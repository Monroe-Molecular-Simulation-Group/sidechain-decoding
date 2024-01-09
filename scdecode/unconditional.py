"""Special input prep, model, and training for glycine (no sidechain) and any N-terminal amino acid (has additional hydrogens)."""

import sys, os
import glob
import argparse
import json
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import openmm as mm
import parmed as pmd
import MDAnalysis as mda
from MDAnalysis.analysis.bat import BAT

import vaemolsim

from . import data_io
from .coord_transforms import get_h_bond_info


def inputs_from_pdb(pdb_file,
                    mod_info,
                    res_name='GLY',
                    bat_atom_str='N,C,CA,HA2,HA3',
                    prep_n_terminal=False
                   ):
    """
    Generates inputs for GLY or N-terminal residues from a pdb file.

    This will only include the BAT coordinates (partial ones for training).
    More generally, this function prepares inputs for residues for which we
    will not be considering the local environment when generating a decoding
    distribution. This is the case for only generating hydrogens for N-terminal
    amino acids, or for glycine. It turns out that GLY will have problems with
    using O, C, and CA as the root atoms. That is because it will not be
    conditioned on the location of N, which will lead to issues. MDAnalysis
    BAT objects, however, cannot handle using C, CA, and N as the root atoms
    because then all dihedrals describing the hydrogens would be improper.

    Parameters
    ----------
    pdb_file : str
        Full path file name for the PDB file to process
    mod_info : str
        A list of residue indices for which mutations and/or modifications, such as
        adding heavy atoms, have been made. These will not be considered for preparing
        training inputs, even if they are glycines.
    res_name : str, default 'GLY'
        Name of residue to prepare inputs for.
    bat_atom_str : str, defualt '@N,C,CA,HA2,HA3'
        Atoms within the residue to consider for BAT analysis. Note this is inclusive,
        not exclusive.
    prep_n_terminal : boolean, default False
        Whether or not we are preparing inputs for N-terminal residues (special case).
        If True, will ignore res_name and consider residues starting with 'N'.

    Returns
    -------
    full_bat : NumPy array
        An array of the full BAT coordinates for each target residue in the protein
        that is within the residue modifications list.
    bat_analysis : MDAnalysis BAT analysis object
        The object used to convert between BAT and XYZ coordinates.
    """
    # For N-terminal residues, know what want for bat_atom_str
    # To save us from having to set that, just set it if prep_n_terminal is True
    if prep_n_terminal:
        # In forcefield xml file, N-terminal have H1, H2, H3, but Parmed uses H, H2, H3
        # In general naming is not consistent - won't matter for forcefield b/c all will have same type
        # But need more inclusive strings for these types of situations
        bat_atom_str = '@N,CA,C,H,H1,H2,H3' 

    # Get pdb id from file
    pdb_id = os.path.split(pdb_file)[-1].split('.pdb')[0]

    # Load structure with openmm first
    mm_pdb = mm.app.PDBFile(pdb_file)

    # Load into ParmEd
    struc = pmd.openmm.load_topology(mm_pdb.topology, xyz=mm_pdb.positions)

    # Follow procedure in openmm.app.ForceField.createSystem() to correctly apply
    # forcefield information to this pdb
    ff = data_io.ff
    ff_data = ff._SystemData(mm_pdb.topology)
    templates_for_residues = ff._matchAllResiduesToTemplates(ff_data, mm_pdb.topology, dict(), False)
    atom_types = [ff_data.atomType[a] for a in mm_pdb.topology.atoms()]
    res_types = [templates_for_residues[r.index].name for r in mm_pdb.topology.residues()]

    # Identify target residues
    # Type should match and include terminal versions of the residue (NXXX or CXXX)
    # Terminal status only impacts backbone, not sidechain atoms, so can decode both
    # But will still exclude modified residues (heavy atoms added)
    # However, must exclude special N-terminal residues NME, NHE, and NPRO
    # Will need special model for NPRO, actually, since only want to decode 2 hydrogens
    target_res_num = []
    for i, r in enumerate(res_types):
        if prep_n_terminal:
            if (r[0] == 'N') and (r not in ['NME', 'NHE', 'NPRO']) and (i not in mod_info):
                target_res_num.append(str(i + 1))
        else:
            if (res_name in r) and (i not in mod_info):
                target_res_num.append(str(i + 1))

    # If have no residues of desired type, stop now
    if len(target_res_num) == 0:
        return None

    # Loop over GLY residues selected and get full BAT information
    full_bat = []

    # Also return BAT analysis object from last target residue
    bat_analysis = None

    for res_num in target_res_num:
    
        this_bat_atoms = struc['(:%s)&(%s)'%(res_num, bat_atom_str)]
        uni = mda.Universe(this_bat_atoms.topology, np.array(this_bat_atoms.coordinates, dtype='float32'))
        if res_name == 'GLY':
            # If working with GLY, want dihedrals to be computed from plane of C, CA, and N
            # To make this work with MDAnalysis BAT analysis, need to rearrange bonds (it's a hack)
            uni.add_bonds([uni.select_atoms('name N or name C')])
            uni.delete_bonds([uni.select_atoms('name N or name CA')])
        bat_analysis = BAT(uni.select_atoms('all'))
        bat_analysis.run()
        this_bat = bat_analysis.results.bat[0] 
        full_bat.append(this_bat)

    # Clean up and return
    full_bat = np.array(full_bat, dtype='float32')
    if prep_n_terminal or (res_name == 'NPRO'):
        # Ensure residue type is N-terminal
        nterm_resname = 'N'+str(bat_analysis._ag[0].resname)
        for a in bat_analysis._ag:
            a.residue.resname = nterm_resname
            # And make sure naming consistent with force field
            if a.name == 'H':
                a.name = 'H1'

    return full_bat, bat_analysis


def build_model(n_atoms, n_H_bonds=0, hidden_dim=100):
    """
    Defines the model that will be used for decoding

    For this model, note the model is unconditional on other atoms present.
    As such, the decoder distribution alone specifies the entire model
    """
    # Define distribution
    # Note setting normal distributions to a std of 0.5 rather than 1
    # This is just so that effectively all of the latent distribution fits inside [-np.pi, np.pi]
    latent_dist = tfp.layers.DistributionLambda(make_distribution_fn=lambda t: tfp.distributions.Blockwise(
                [tfp.distributions.Normal(loc=tf.zeros((tf.shape(t)[0],)),
                                          scale=0.5*tf.ones((tf.shape(t)[0],)))] * (2 * n_atoms - n_H_bonds)
                + [tfp.distributions.VonMises(loc=tf.zeros((tf.shape(t)[0],)), concentration=tf.ones((tf.shape(t)[0],)))] * n_atoms)
                                               ) # Bonds and angles modeled as normal distributions, torsions as von Mises
    flow = vaemolsim.flows.RQSSplineMAF(num_blocks=3, # Three RQS flows, middle with "random" ordering
                                        order_seed=42, # Setting seed makes order deterministic (so can load weights)
                                        rqs_params={'bin_range': [-np.pi, np.pi], # Range should work for bonds and angles, too
                                                    'num_bins': 20, # Can place spline knot every ~0.314 units
                                                    'hidden_dim': hidden_dim,
                                                    'conditional': False},
                                        batch_norm=False, # Batch norm messes with fixed domain for periodic flows
                                       )

    # Here, the decoder distribution is the full model (no embedding or mapping, etc.)
    model = vaemolsim.models.FlowModel(flow, latent_dist)
    _ = model.flowed_dist.flow(tf.ones([1, n_atoms * 3 - n_H_bonds])) # Build flow

    return model


def train_model(read_dir='./', save_dir='./', save_name='sidechain', constrain_H_bonds=False):
    """
    Creates and trains a model for decoding.
    """

    # Read in data, here .npy files with BAT coordinates
    # Since numpy data, will split out validation set when specifying model fit
    files = glob.glob('%s/*.npy'%read_dir)
    # Exclude root atom BAT coords - root atom coords are first 9
    train_data = np.vstack([np.load(f)[:, 9:] for f in files]).astype('float32') 

    # Set up model
    # First need number of degrees of freedom to predict from BAT analysis object
    bat_obj_file = glob.glob('%s/*.pkl'%read_dir)[0]
    with open(bat_obj_file, 'rb') as f:
        bat_obj = pickle.load(f)
    n_atoms = len(bat_obj._torsions) # Will also be number of bonds, angles, and torsions

    # If masking out bonds involving hydrogens
    if constrain_H_bonds:
        h_inds, non_h_inds, h_bond_lengths = get_h_bond_info(bat_obj)
        n_H_bonds = len(h_inds)
        train_data = train_data[:, non_h_inds]
    else:
        n_H_bonds = 0

    model = build_model(n_atoms, n_H_bonds=n_H_bonds)

    # Set optimizer and compile
    model.compile(tf.keras.optimizers.Adam(),
                  loss=vaemolsim.losses.LogProbLoss(),
                 )

    # Any callbacks needed? Shouldn't really need annealing
    callback_list = [tf.keras.callbacks.ModelCheckpoint(os.path.join(save_dir, '%s_decoder'%save_name, '%s_weights.ckpt'%save_name),
                                                        monitor='val_loss',
                                                        mode='min',
                                                        save_best_only=True,
                                                        save_weights_only=True
                                                       ),
                     tf.keras.callbacks.TerminateOnNaN(),
                    ]

    # Fit the model
    history = model.fit(x=train_data, y=train_data, epochs=10, verbose=2, callbacks=callback_list,
                        batch_size=64, validation_split=0.1, shuffle=True,
                       )

    print(model.summary())

    # Save history
    np.savez(os.path.join(save_dir, '%s_history.npz'%save_name), **history.history)


def main_gen_input(arg_list):
    """
    Generates training inputs.
    """
    parser = argparse.ArgumentParser(prog='unconditional.main_gen_input.py',
                                     description='Generates training inputs or trains models for unconditional cases.',
                                    )
    parser.add_argument('mod_file', help="json file containing dictionary of residue modifications for each pdb")
    parser.add_argument('res_type', help="residue type to prepare inputs for")
    parser.add_argument('--read_dir', '-r', default='./', help="directory to read files from")
    parser.add_argument('--save_dir', '-s', default='./', help="directory to save outputs to")
    parser.add_argument('--bat_atom_str', default='@N,C,CA,HA2,HA3', help="selection string for BAT analysis atoms")
    parser.add_argument('--n_terminal', action='store_true', help="if specified, handles special case of N-terminal residues")

    args = parser.parse_args(arg_list)

    # Get dictionary of modifications
    with open(args.mod_file, 'r') as f:
        mod_info = json.load(f)

    # Get a list of all pdb files to process
    pdb_list = glob.glob(os.path.join(args.read_dir, '*.pdb'))
    print('Found %i pdb files to process.'%len(pdb_list))

    # Loop over pdb files and train
    saved_bat_obj = False
    for p in pdb_list:
        pdb_id = os.path.split(p)[-1].split('.pdb')[0]
        this_mod = mod_info[pdb_id]
       
        try:
            inputs = inputs_from_pdb(p, this_mod, res_name=args.res_type,
                                     bat_atom_str=args.bat_atom_str, prep_n_terminal=args.n_terminal)
        except Exception as exc:
            print('On file %s, failed with exception:\n%s'%(p, str(exc)))
            inputs = None

        # Check to make sure had residues to work with
        if inputs is not None:

            # Specifically save the full set of BAT coordinates
            np.save(os.path.join(args.save_dir, '%s_%s_full_BAT.npy'%(pdb_id, args.res_type)),
                    inputs[0],
                    allow_pickle=False
                   )

            # Pickle the most recent BAT object
            # Only need to do once per residue type
            # So do as soon as inputs not None
            if not saved_bat_obj:
                pickle.dump(inputs[1], open(os.path.join(args.save_dir, '%s_BAT_object.pkl'%(args.res_type)), 'wb'))
                saved_bat_obj = True


def main_train(arg_list):
    """
    Builds and trains model on data.
    """
    parser = argparse.ArgumentParser(prog='unconditional.main_train.py',
                                     description='Trains a decoding model (unconditional flow only).',
                                    )
    parser.add_argument('res_type', help="residue type to prepare inputs for")
    parser.add_argument('--read_dir', '-r', default='./', help="directory to read files from")
    parser.add_argument('--save_dir', '-s', default='./', help="directory to save outputs to")
    parser.add_argument('--h_bonds', action='store_true', help='whether or not to constrain bonds with hydrogens')

    args = parser.parse_args(arg_list)

    train_model(read_dir=args.read_dir,
                save_dir=args.save_dir,
                save_name=args.res_type,
                constrain_H_bonds=args.h_bonds,
               )

    
if __name__ == '__main__':
    if sys.argv[1] == 'gen_input':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        main_gen_input(sys.argv[2:])
    elif sys.argv[1] == 'train':
        main_train(sys.argv[2:])
    else:
        print("Argument \'%s\' unrecognized. For the first argument, select either \'gen_input\' or \'train\'"%sys.argv[1])
