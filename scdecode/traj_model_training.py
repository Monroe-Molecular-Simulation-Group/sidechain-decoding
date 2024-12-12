
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
# Separating into two losses is tricky because then need two model outputs
# With the decoder output, that results in a subtle error when running in graph mode that I couldn't fix
# So just use this and a metric, which is a little bit of extra computation, but gives break down of loss
# To mask hydrogen bonds WITHOUT any CG penalty, use this loss but set one_over_cg_var to zero
class LogProbPenalizedCGLoss(tf.keras.losses.Loss):
    """
    A loss to enforce mapping of output to CG coordinates in addition to log probability.
    """

    def __init__(self, bat_obj, one_over_cg_var=4.0, mask_H=False, n_samples=10, name='log_prob_cg_loss', **kwargs):
        """
        Creates loss object.

        Parameters
        ----------
        bat_obj : MDAnalysis BAT analysis object
            BAT analysis object defining the transformation.
        one_over_cg_var : float, default 4.0
            Reciprocal variance for distribution of CG coordinates from reference.
            If set to 0.0, ignores CG penalty term.
        mask_H : bool, default False
            Whether or not to define mask over bonds involving hydrogens.
        n_samples : int, default 10
            Number of samples to draw for computing the CG bead location
        """

        super(LogProbPenalizedCGLoss, self).__init__(name=name, **kwargs)

        self.bat_obj = bat_obj

        self.one_over_cg_var = tf.Variable(one_over_cg_var, trainable=False)

        self.n_samples = n_samples

        if mask_H:
            h_inds, non_h_inds, h_bond_lengths = get_h_bond_info(bat_obj)
        else:
            h_inds = []
            non_h_inds = list(range(len(bat_obj._torsions)*3))
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
        n_batch = tf.shape(targets)[0]

        # Need to pick apart full BAT to compute log-probability correctly
        bat = tf.gather(full_bat[:, 9:], self.non_h_inds, axis=-1)
        log_prob = -decoder.log_prob(bat)

        if self.one_over_cg_var == 0.0:
            # Use if-statement to save time, but also because sampling step is less numerically stable
            # (has some chance of producing NaN if the variances predicted for von Mises distributions become too small)
            cg_penalty = 0.0
        else:
            # Now need to sample from the distribution and check CG positions
            # sample will be of shape (n_samples, N_batch, N_DOFs)
            sample = decoder.sample(self.n_samples)

            # Insert H-bond values
            h_bond_vals = tf.tile(tf.reshape(self.h_bond_lengths, (1, 1, -1)), (self.n_samples, n_batch, 1))
            full_sample = tf.transpose(tf.dynamic_stitch([self.non_h_inds, self.h_inds],
                                                    [tf.transpose(sample), tf.transpose(h_bond_vals)])
                                      )
        
            # Combine predicted values with root positions in target
            full_sample = tf.concat([tf.tile(tf.expand_dims(full_bat[:, :9], axis=0), (self.n_samples, 1, 1)),
                                     full_sample],
                                    axis=-1)

            # Obtain XYZ indices
            # To make this work with n_samples samples, need to flatten, apply, then reshape back
            # (after compute CG locations)
            full_sample = tf.reshape(full_sample, (self.n_samples*n_batch, -1))
            xyz_sample = bat_cartesian_tf(full_sample, self.bat_obj)

            # Compute location of CG site
            cg_sample = tf.reduce_sum(self.mass_weights * xyz_sample, axis=1)
            cg_sample = tf.reshape(cg_sample, (self.n_samples, n_batch, -1))

            cg_ref = tf.tile(tf.expand_dims(targets[:, -3:], axis=0), (self.n_samples, 1, 1)), 
            # Assuming Gaussian distribution, enforce sampled CG close to reference
            cg_penalty = tf.reduce_mean(tf.reduce_sum(self.one_over_cg_var * 0.5 * (cg_sample - cg_ref)**2, axis=-1), axis=0)
            # Or try a Laplace distribution instead
            # cg_penalty = tf.reduce_mean(tf.reduce_sum(self.one_over_cg_var * tf.math.abs(cg_sample - cg_ref), axis=-1), axis=0)

        return log_prob + cg_penalty


class CGPenaltyAnnealing(tf.keras.callbacks.Callback):
    """
    Adjusts the CG penalty on a linear schedule (to be used with LogProbPenalizedCGLoss)
    """

    def __init__(self, start_epoch, end_epoch, start_val, end_val):
        super().__init__()
        
        if start_epoch >= end_epoch:
            raise ValueError('start_epoch (%i) must be smaller than end_epoch (%i)'%(start_epoch, end_epoch))

        self.start_epoch = start_epoch - 1 # Counting of epochs starts at 0 but displays starting at 1
        self.end_epoch = end_epoch - 1
        self.start_val = start_val
        self.end_val = end_val
        self.rate = (end_val - start_val) / (end_epoch - start_epoch)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch <= self.start_epoch:
            this_val = self.start_val
        elif epoch < self.end_epoch:
            this_val = self.rate * (epoch - self.start_epoch) + self.start_val
        else:
            this_val  = self.end_val
        self.model.loss.one_over_cg_var.assign(this_val)
        print("\nEpoch %i: Set reciprocal CG penalty variance to %6.4f."%(epoch + 1, this_val))


class MeanLogProb(tf.keras.metrics.Metric):
    """
    To enable decomposition of loss, build in log-prob loss as metric.

    This breaks out the log-prob component of LogProbPenalizedCGLoss.
    """

    def __init__(self, non_h_inds, name='mean_log_prob', **kwargs):
        super(MeanLogProb, self).__init__(name=name, **kwargs)
        self.non_h_inds = non_h_inds
        self.log_prob_loss = self.add_weight(name='log_prob', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, targets, decoder, sample_weight=None):
        full_bat = targets[:, :-3]

        # Need to pick apart full BAT to compute log-probability correctly
        bat = tf.gather(full_bat[:, 9:], self.non_h_inds, axis=-1)
        log_prob_loss = -decoder.log_prob(bat)

        if sample_weight is not None:
            sample_weight = tf.broadcast_to(sample_weight, log_prob_loss.shape)
            log_prob_loss = tf.multiply(log_prob_loss, sample_weight)

        self.log_prob_loss.assign_add(tf.reduce_sum(log_prob_loss))

        self.count.assign_add(tf.cast(tf.shape(targets)[0], self.dtype))

    def result(self):
        return self.log_prob_loss / self.count


def build_model(n_atoms, n_H_bonds=0, embed_dim=None, n_particles=150, hidden_dim=100):
    """
    Defines the model that will be used for side-chain decoding
    """
    # If embedding dimension not specified, select based on the number of heavy atoms
    # This is the number of atoms minus the number of bonds involving hydrogens
    # Multiply the number of heavy atoms (+1 so don't have zero for alanine) by 10
    if embed_dim is None:
        n_heavy = n_atoms - n_H_bonds
        embed_dim = 10 * (n_heavy + 1)
    else:
        embed_dim = int(embed_dim)

    # Define distance-based embedding
    mask_dist = vaemolsim.mappings.DistanceSelection(8.0, n_particles) # 8 Angstrom cutoff, no more then n_particles included
    particle_embed = vaemolsim.mappings.ParticleEmbedding(embed_dim, hidden_dim=embed_dim)
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


def train_model(read_dir='./',
                save_dir='./',
                save_name='sidechain',
                include_cg_target=False,
                constrain_H_bonds=False,
                num_epochs=20,
               ):
    """
    Creates and trains a model for decoding a sidechain.
    """

    # Read in data, randomly splitting based on .tfrecord files
    # Means will not be exactly 90/10, but will be close-ish
    files = glob.glob('%s/*.tfrecord'%read_dir)
    # train_files = []
    # val_files = []
    # for i, f in enumerate(files):
    #     if i % 10 == 0:
    #         val_files.append(f)
    #     else:
    #         train_files.append(f)
    # train_dset = read_dataset(train_files, include_cg_target=include_cg_target)
    # val_dset = read_dataset(val_files, include_cg_target=include_cg_target)
    dset = read_dataset(files, include_cg_target=include_cg_target)
    val_dset = dset.shard(num_shards=10, index=0)
    train_dset = dset.shard(num_shards=10, index=1)
    for k in range(2, 10):
        train_dset = train_dset.concatenate(dset.shard(num_shards=10, index=k))

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
        loss = LogProbPenalizedCGLoss(bat_obj, mask_H=constrain_H_bonds, n_samples=100)
        metrics = MeanLogProb(loss.non_h_inds)
    else:
        loss = vaemolsim.losses.LogProbLoss()
        metrics = None
    model.compile(tf.keras.optimizers.Adam(),
                  loss=loss,
                  metrics=metrics,
                 )

    # Any callbacks needed? Shouldn't really need annealing
    callback_list = [tf.keras.callbacks.ModelCheckpoint(os.path.join(save_dir, '%s_decoder'%save_name, '%s_weights.ckpt'%save_name),
                                                        monitor='val_loss',
                                                        mode='min',
                                                        save_best_only=True, # If annealing CG penalty, set to False
                                                        save_weights_only=True
                                                       ),
                     tf.keras.callbacks.TerminateOnNaN(),
                    ]

    if include_cg_target:
        # Current default is no annealing (same start and end value)
        callback_list.append(CGPenaltyAnnealing(5, 10, 0.0, 0.0))

    # Fit the model
    history = model.fit(train_dset, epochs=num_epochs, validation_data=val_dset, verbose=2, callbacks=callback_list)

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

    # Will define dictionary of training epochs based on residue type
    # Some residues have fewer samples, so benefit from more training
    # Others are simply more complicated and also benefit from more training
    # 20 epochs will be "standard"
    n_epochs_dict = {'ALA': 20,
                     'ARG': 40, # Larger, more flexible
                     'ASH': 2000, # order 1e2 training samples instead of order 1e5
                     'ASN': 20,
                     'ASP': 30, # Charges and strongly directional interactions
                     'CYS': 100, # order 1e4 rather than 1e5 samples
                     'GLH': 2000, # order 1e2 training samples instead of 1e5
                     'GLN': 20,
                     'GLU': 30, # Charges and strongly directional interactions
                     'HID': 30,  # Bulky and polar
                     'HIE': 300, # order 1e4 rather than 1e5 samples
                     'HIP': 2000, # order 1e3 rather than 1e5 samples
                     'ILE': 20,
                     'LEU': 20,
                     'LYS': 30, # Larger, more flexible
                     'MET': 30, # Somewhat fewer samples, but also larger with less common chemistry
                     'PHE': 40, # Bulkier sidechain
                     'PRO': 30, # Unusually rigid
                     'SER': 20,
                     'THR': 20,
                     'TRP': 100, # Bulkiest sidechain, fewer training examples
                     'TYR': 40, # Bulky sidechain
                     'VAL': 20,
                    }

    train_model(read_dir=args.read_dir,
                save_dir=args.save_dir,
                save_name=args.res_type,
                include_cg_target=args.cg_target,
                constrain_H_bonds=args.h_bonds,
                num_epochs=n_epochs_dict[args.res_type],
               )


if __name__ == "__main__":
    main(sys.argv[1:])
