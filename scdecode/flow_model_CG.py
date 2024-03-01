"""
Trains a normalizing flow model (Boltzmann Generator) for the CG representation from an all-atom trajectory.
"""
import sys, os
import pickle
import argparse

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import vaemolsim

import parmed as pmd
import MDAnalysis as mda
from MDAnalysis.analysis.bat import BAT

from scdecode import analysis_tools, coord_transforms


def build_flow_model(n_atoms, n_H_bonds=0, hidden_dim=100):
    """
    Defines the model to be used for effectively learning to sample a CG system
    """

    # Define starting distribution - will be static as standard normals and standard VonMises
    # Will exclude first 6 DoFs in BAT coordinates (rigid translation and rotation) but include the first two bonds and angle 
    # of the root atoms, which adds 3 degrees of freedom to the normal distributions
    # So n_atoms is really the number of non-root atoms
    latent_dist = tfp.layers.DistributionLambda(make_distribution_fn=lambda t: tfp.distributions.Blockwise(
        [tfp.distributions.Normal(loc=tf.zeros((tf.shape(t)[0],)),
                                  scale=0.5*tf.ones((tf.shape(t)[0],)))] * (3 + 2 * n_atoms - n_H_bonds)
        + [tfp.distributions.VonMises(loc=tf.zeros((tf.shape(t)[0],)),
                                  concentration=tf.ones((tf.shape(t)[0],)))] * n_atoms)
        )

    # Define the flow
    flow = vaemolsim.flows.RQSSplineMAF(num_blocks=5,
                                        order_seed=42,
                                        rqs_params={'bin_range': [-np.pi, np.pi],
                                                    'num_bins': 20,
                                                    'hidden_dim': hidden_dim,
                                                    'conditional': False},
                                        batch_norm=False,
                                        )
    model = vaemolsim.models.FlowModel(flow, latent_dist)
    _ = model.flowed_dist.flow(tf.ones([1, 3 + n_atoms * 3 - n_H_bonds])) # Build flow

    return model


def load_training_data_from_aa_traj(pdb_file, traj_file, out_name=None):
    """
    Loads all-atom trajectory and converts to BAT coordinates of CG representation for training.
    """
    if out_name is None:
        out_name = pdb_file.split('.pdb')[0].split('/')[-1]

    # Load via openmm to make sure topology correctly created (with right residue types, etc.)
    pdb_obj, sim_obj = analysis_tools.sim_from_pdb(pdb_file)
    pmd_struc = pmd.openmm.load_topology(pdb_obj.topology, system=sim_obj.system, xyz=pdb_obj.positions)
    uni = mda.Universe(pmd_struc.topology, traj_file)

    # Convert to a CG trajectory inside a ParmEd structure
    cg_struc = analysis_tools.create_cg_structure(pmd_struc, mda_uni=uni)

    # Save a pdb CG structure
    cg_struc.save('%s_CG.pdb'%out_name, coordinates=cg_struc.get_coordinates()[0, ...], renumber=True, overwrite=True)

    # And go back to new MDAnalysis universe for converting to BAT coordinates
    cg_uni = mda.Universe(cg_struc.topology, np.array(cg_struc.get_coordinates(), dtype='float32'))
    bat_analysis = BAT(cg_uni.select_atoms('all'))
    bat_analysis.run()
    
    # Save BAT analysis object as well as BAT trajectory
    pickle.dump(bat_analysis, open('%s_CG_BAT_object.pkl'%(out_name), 'wb'))
    np.save('%s_CG_BAT_traj.npy'%(out_name), bat_analysis.results.bat, allow_pickle=False)

    # And return BAT object, which includes coordinates for training
    return bat_analysis


def train_flow_model(pdb_file, traj_file, save_dir='./', constrain_H_bonds=False):
    """
    Loads training data, creates and trains a flow model for CG configurations from an all-atom trajectory
    """
    out_name = pdb_file.split('.pdb')[0].split('/')[-1]

    # Load in training data
    bat_obj = load_training_data_from_aa_traj(pdb_file, traj_file, out_name=os.path.join(save_dir, out_name))

    # Exclude rigid-body translational and rotational degrees of freedom
    train_data = bat_obj.results.bat[:, 6:]

    # Get number of (non-root) atoms and information on bonds involving hydrogens
    n_atoms = len(bat_obj._torsions)
    if constrain_H_bonds:
        h_inds, non_h_inds, h_bond_lengths = coord_transforms.get_h_bond_info(bat_obj)
        # Above assumes excluding first 9 BAT DoFs, so must add 3 to all indices
        # Root atoms should be heavy atoms, not hydrogens
        h_inds = [i + 3 for i in h_inds]
        non_h_inds = [0, 1, 2] + [i + 3 for i in non_h_inds]
        n_H_bonds = len(h_inds)
        train_data = train_data[:, non_h_inds]
    else:
        n_H_bonds = 0

    model = build_flow_model(n_atoms, n_H_bonds=n_H_bonds)

    # Set optimizer and compile
    model.compile(tf.keras.optimizers.Adam(),
                  loss=vaemolsim.losses.LogProbLoss()
                  )

    # Set up callbacks
    callback_list = [tf.keras.callbacks.ModelCheckpoint(os.path.join(save_dir, '%s_CG_flow'%out_name, '%s_CG_flow_weights.ckpt'%out_name),
                                                        monitor='val_loss',
                                                        model='min',
                                                        save_best_only=True,
                                                        save_weights_only=True,
                                                        ),
                     tf.keras.callbacks.TerminateOnNaN(),
                     ]

    # Fit the model
    history = model.fit(x=train_data, y=train_data, epochs=100, verbose=2, callbacks=callback_list,
                        batch_size=256, validation_split=0.1, shuffle=True,
                       )

    print(model.summary())

    # Save history
    np.savez(os.path.join(save_dir, '%s_CG_flow_history.npz'%out_name), **history.history)


def sample_flow_model(bat_obj, model_dir, save_dir='./', n_samples=1000, constrain_H_bonds=False):
    # will want to load in model based on bat_obj like when training
    # Can put together dummy set of inputs in right shape to call predict
    # Once have all samples from predict, can call log_prob on all of them with model.flowed_dist.log_prob()
    # Save the log-probabilities
    # Will want to then fill in h-bonds, save BAT sample, combine with first 6 DoFs, convert to XYZ and save
    # For xyz, save in trajectory format like .nc for compatibility with decoding in full_protein_decoding
    pass 


def main_train(arg_list):
    """
    Builds and trains model after loading training data.
    """
    parser = argparse.ArgumentParser(prog='train_CG_flow.py train',
                                     description='Trains flow model for CG BAT coordinates from all-atom trajectory'
                                     )
    parser.add_argument('pdb_file', help='pdb or structure file')
    parser.add_argument('traj_file', help='trajectory file')
    parser.add_argument('--save_dir', '-s', default='./', help='directory to save outputs to')
    parser.add_argument('--h_bonds', action='store_true', help='whether or not to constrain bonds with hydrogens')

    args = parser.parse_args(arg_list)

    train_flow_model(args.pdb_file,
                     args.traj_file,
                     save_dir=args.save_dir,
                     constrain_H_bonds=args.h_bonds,
                    )

def main_sample(arg_list):
    """
    Samples a CG trajectory from a trained model
    """
    parser = argparse.ArgumentParser(prog='train_CG_flow.py sample',
                                     description='Loads a trained model and samples from it'
                                     )
    parser.add_argument('bat_file', help='path to BAT analysis object file')
    parser.add_argument('model_dir', help='directory where model is saved')
    parser.add_argument('--save_dir', '-s', default='./', help='directory to save outputs to')
    parser.add_argument('--n_samples', '-n', default=100000, help='number of samples to draw')
    parser.add_argument('--h_bonds', action='store_true', help='whether or not to constrain bonds with hydrogens')

    args = parser.parse_args(arg_list)

    sample_flow_model(args.pdb_file,
                     args.traj_file,
                     save_dir=args.save_dir,
                     constrain_H_bonds=args.h_bonds,
                    )


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        main_train(sys.argv[2:])
    elif sys.argv[1] == 'sample':
        main_sample(sys.argv[2:])
    else:
        print("Argument \'%s\' not recognized. For the first argument select \'train\' or \'sample\'."%sys.argv[1])
