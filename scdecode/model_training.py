
import sys, os
import argparse
import glob
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import vaemolsim

from . import data_io

from .data_io import read_dataset
from .coord_transforms import bat_cartesian_tf, get_h_bond_info


# Need custom loss function to help enforce CG location
class LogProbPenalizedCGLoss(tf.keras.losses.Loss):
    """
    A loss to enforce mapping of output to CG coordinates in addition to log probability.
    """

    def __init__(self, bat_obj, cg_var=1.0, mask_H=False, name='log_prob_cg_loss', **kwargs):
        """
        Creates loss object.

        Parameters
        ----------
        bat_obj : MDAnalysis BAT analysis object
            BAT analysis object defining the transformation.
        cg_var : float, default 1.0
            Variance for distribution of CG coordinates from reference.
        mask_H : bool, default False
            Whether or not to define mask over bonds involving hydrogens.
        """

        super(LogProbPenalizedCGLoss, self).__init__(name=name, **kwargs)

        self.bat_obj = bat_obj

        self.cg_var = cg_var

        if mask_H:
            h_inds, non_h_inds, h_bond_lengths = get_h_bond_info(bat_obj)
        else:
            h_inds = []
            non_h_inds = list(range(len(bat_obj._torsions)))
            h_bond_lengths = []
        self.h_inds = h_inds
        self.non_h_inds = non_h_inds
        self.h_bond_lengths = tf.convert_to_tensor(h_bond_lengths)

        # Also need masses of atoms contributing to sidechain CG position
        # That will include all but the first 2 root atoms and the HA atom
        # (assuming that root atoms are C, CA, CB or N, CA, CB)
        # Set masses of non-sidechain atoms to 0.0 so do not contribute
        masses = []
        for a in bat_obj.atoms:
            if a.name not in data_io.backbone_atoms[1:].split(','):
                masses.append(a.mass)
            else:
                masses.append(0.0)
        mass_weights = np.array(masses) / np.sum(masses)
        self.mass_weights = tf.reshape(tf.cast(mass_weights, tf.float32), (1, -1, 1))

    def call(self, targets, decoder):
        """
        Computes the log-probability of samples under a provided tfp.distribution object.
        And adds on a penalty for being far from the CG reference when sample from the distribution.

        Parameters
        ----------
        targets : tf.Tensor
            Tensor with shape (N_batch, N_BAT+3), with full BAT coordinates up to the last 3
            columns, with those last 3 being the CG reference location.
        decoder : tfp.distributions object
            An object representing model probability density (must have a log_prob() method).

        Returns
        -------
        loss : tf.Tensor
            Negative log-probability of samples under decoder, without taking average
            over batch (i.e, it returns the per-sample loss). Adds this to penalization
            of CG coordinate of sample from the reference.
        """
        # Define inputs
        # List of inputs is more intuitive, but is not possible with how tf.keras.Model handles loss inputs
        full_bat = targets[:, :-3]
        cg_ref = targets[:, -3:]

        # Need to pick apart full BAT to compute log-probability correctly
        bat = tf.gather(full_bat[:, 9:], self.non_h_inds, axis=-1)
        log_prob = -decoder.log_prob(bat)

        # Now need to sample from the distribution and check CG positions
        sample = decoder.sample()

        # Insert H-bond values
        h_bond_vals = tf.tile(tf.reshape(self.h_bond_lengths, (1, -1)), (tf.shape(sample)[0], 1))
        full_sample = tf.transpose(tf.dynamic_stitch([self.non_h_inds, self.h_inds],
                                                [tf.transpose(sample), tf.transpose(h_bond_vals)])
                                  )
        
        # Combine predicted values with root positions in target
        full_sample = tf.concat([full_bat[:, :9], full_sample], axis=-1)

        # Obtain XYZ indices
        xyz_sample = bat_cartesian_tf(full_sample, self.bat_obj)

        # Compute location of CG reference site
        cg_sample = tf.reduce_sum(self.mass_weights * xyz_sample, axis=1)

        # Assuming Gaussian distribution, enforce sampled CG close to reference
        cg_penalty = tf.reduce_sum((cg_sample - cg_ref)**2 / (2.0 * self.cg_var), axis=-1)

        return log_prob + cg_penalty


def build_model(n_atoms, n_H_bonds=0, embed_dim=20, hidden_dim=100):
    """
    Defines the model that will be used for side-chain decoding
    """
    # Define distance-based embedding
    mask_dist = vaemolsim.mappings.DistanceSelection(5.0, 50) # 5 Angstrom cutoff, no more then 50 particles included
    particle_embed = vaemolsim.mappings.ParticleEmbedding(embed_dim)
    mask_and_embed = vaemolsim.mappings.LocalParticleDescriptors(mask_dist, particle_embed)

    # Define distribution (and mapping to it)
    latent_dist = vaemolsim.dists.IndependentBlockwise(n_atoms * 3 - n_H_bonds,
                   [tfp.distributions.Normal] * (2 * n_atoms - n_H_bonds) + [tfp.distributions.VonMises] * n_atoms,
                  ) # Bonds and angles modeled as normal distributions, torsions as von Mises
    flow = vaemolsim.flows.RQSSplineMAF(num_blocks=3, # Three RQS flows, middle with "random" ordering
                                        order_seed=42, # Setting seed makes order deterministic (so can load weights)
                                        rqs_params={'bin_range': [-np.pi, np.pi], # Range should work for bonds and angles, too
                                                    'num_bins': 20, # Can place spline knot every ~0.314 units
                                                    'hidden_dim': hidden_dim,
                                                    'conditional': True,
                                                    'conditional_event_shape': embed_dim},
                                        batch_norm=False, # Batch norm messes with fixed domain for periodic flows
                                       ) 
    decoder_dist = vaemolsim.dists.FlowedDistribution(flow, latent_dist)
    _ = decoder_dist.flow(tf.ones([1, n_atoms * 3 - n_H_bonds]),
                          conditional_input=tf.ones([1, embed_dim])
                         ) # Build flow
    map_embed_to_dist = vaemolsim.mappings.FCDeepNN(decoder_dist.params_size(),
                                                    hidden_dim=hidden_dim,
                                                    batch_norm=False, # Not deep enough to benefit
                                                   )
    decoder = vaemolsim.models.MappingToDistribution(decoder_dist, mapping=map_embed_to_dist, name='decoder')

    # Finish full model
    model = vaemolsim.models.BackmappingOnly(mask_and_embed, decoder)
    return model


def train_model(read_dir='./', save_dir='./', save_name='sidechain', include_cg_target=False, constrain_H_bonds=False):
    """
    Creates and trains a model for decoding a sidechain.
    """

    # Read in data, randomly splitting based on .tfrecord files
    # Means will not be exactly 90/10, but will be close-ish
    files = glob.glob('%s/*.tfrecord'%read_dir)
    train_files = []
    val_files = []
    for i, f in enumerate(files):
        if i % 10 == 0:
            val_files.append(f)
        else:
            train_files.append(f)
    train_dset = read_dataset(train_files, include_cg_target=include_cg_target)
    val_dset = read_dataset(val_files, include_cg_target=include_cg_target)

    # Should shuffle and batch training dataset (also set up prefetching)
    # For validation, just batch and prefetch
    train_dset = train_dset.shuffle(1000).ragged_batch(64, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    val_dset = val_dset.ragged_batch(64).prefetch(tf.data.AUTOTUNE)

    # Set up model
    # First need number of degrees of freedom to predict from BAT analysis object
    bat_obj_file = glob.glob('%s/*.pkl'%read_dir)[0]
    with open(bat_obj_file, 'rb') as f:
        bat_obj = pickle.load(f)
    n_atoms = len(bat_obj._torsions) # Will also be number of bonds, angles, and torsions
    n_H_bonds = 0
    if constrain_H_bonds:
        for i, a in enumerate(bat_obj._ag1.atoms):
            if a.element == 'H':
                n_H_bonds += 1

    model = build_model(n_atoms, n_H_bonds=n_H_bonds)

    # Set optimizer and compile
    if include_cg_target:
        loss = LogProbPenalizedCGLoss(bat_obj, mask_H=constrain_H_bonds)
    else:
        loss = vaemolsim.losses.LogProbLoss()
    model.compile(tf.keras.optimizers.Adam(),
                  loss=loss,
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
    history = model.fit(train_dset, epochs=10, validation_data=val_dset, verbose=2, callbacks=callback_list)

    print(model.summary())

    # Save history
    np.savez(os.path.join(save_dir, '%s_history.npz'%save_name), **history.history)


def main(arg_list):
    parser = argparse.ArgumentParser(prog='model_training.py',
                                     description='Trains a sidechain decoding model.',
                                    )
    parser.add_argument('res_type', help="residue type to prepare inputs for")
    parser.add_argument('--read_dir', '-r', default='./', help="directory to read files from")
    parser.add_argument('--save_dir', '-s', default='./', help="directory to save outputs to")
    # Automatically sets default to False
    parser.add_argument('--cg_target', action='store_true', help='whether or not to penalize CG bead distance') 
    parser.add_argument('--h_bonds', action='store_true', help='whether or not to constrain bonds with hydrogens')

    args = parser.parse_args(arg_list)

    train_model(read_dir=args.read_dir,
                save_dir=args.save_dir,
                save_name=args.res_type,
                include_cg_target=args.cg_target,
                constrain_H_bonds=args.h_bonds,
               )


if __name__ == "__main__":
    main(sys.argv[1:])
