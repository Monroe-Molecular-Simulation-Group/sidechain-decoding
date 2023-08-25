
import sys, os
import argparse
import glob
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import vaemolsim

from .data_io import read_dataset


def build_model(n_atoms, embed_dim=20, hidden_dim=100):
    """
    Defines the model that will be used for side-chain decoding
    """
    # Define distance-based embedding
    mask_dist = vaemolsim.mappings.DistanceSelection(5.0, 50) # 5 Angstrom cutoff, no more then 50 particles included
    particle_embed = vaemolsim.mappings.ParticleEmbedding(embed_dim)
    mask_and_embed = vaemolsim.mappings.LocalParticleDescriptors(mask_dist, particle_embed)

    # Define distribution (and mapping to it)
    latent_dist = vaemolsim.dists.IndependentBlockwise(n_atoms * 3,
                   [tfp.distributions.Normal] * (2 * n_atoms) + [tfp.distributions.VonMises] * n_atoms,
                  ) # Bonds and angles modeled as normal distributions, torsions as von Mises
    flow = vaemolsim.flows.RQSSplineMAF(num_blocks=3, # Three RQS flows, middle with "random" ordering
                                        order_seed=42, # Setting seed makes order deterministic (so can load weights)
                                        rqs_params={'bin_range': [-np.pi, np.pi], # Range should work for bonds and angles, too
                                                    'num_bins': 20, # Can place spline knot every ~0.314 units
                                                    'hidden_dim': hidden_dim,
                                                    'conditional': True,
                                                    'conditional_event_shape': embed_dim},
                                        batch_norm=True,
                                       ) 
    decoder_dist = vaemolsim.dists.FlowedDistribution(flow, latent_dist)
    _ = decoder_dist.flow(tf.ones([1, n_atoms * 3]),
                          conditional_input=tf.ones([1, embed_dim])
                         ) # Build flow
    map_embed_to_dist = vaemolsim.mappings.FCDeepNN(decoder_dist.params_size(),
                                                    hidden_dim=hidden_dim,
                                                    batch_norm=True,
                                                   )
    decoder = vaemolsim.models.MappingToDistribution(decoder_dist, mapping=map_embed_to_dist, name='decoder')

    # Finish full model
    model = vaemolsim.models.BackmappingOnly(mask_and_embed, decoder)
    return model


def train_model(read_dir='./', save_dir='./', save_name='sidechain_decoder'):
    """
    Creates and trains a model for decoding a sidechain.
    """

    # Read in data, randomly splitting based on .tfrecord files
    # Means will not be exactly 90/10, but will be closish
    files = glob.glob('%s/*.tfrecord'%read_dir)
    train_files = []
    val_files = []
    for i, f in enumerate(files):
        if i % 10 == 0:
            val_files.append(f)
        else:
            train_files.append(f)
    train_dset = read_dataset(train_files)
    val_dset = read_dataset(val_files)

    # Should shuffle and batch training dataset (also set up prefetching)
    # For validation, just batch and prefetch
    train_dset = train_dset.shuffle(1000).ragged_batch(200).prefetch(tf.data.AUTOTUNE)
    val_dset = val_dset.ragged_batch(200).prefetch(tf.data.AUTOTUNE)

    # Set up model
    # First need number of degrees of freedom to predict from BAT analysis object
    bat_obj_file = glob.glob('%s/*.pkl'%read_dir)[0]
    with open(bat_obj_file, 'rb') as f:
        bat_obj = pickle.load(f)
    n_atoms = len(bat_obj._torsions) # Will also be number of bonds, angles, and torsions
    model = build_model(n_atoms)

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
    history = model.fit(train_dset, epochs=10, validation_data=val_dset, callbacks=callback_list)

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

    args = parser.parse_args(arg_list)

    train_model(read_dir=args.read_dir, save_dir=args.save_dir, save_name=args.res_type)


if __name__ == "__main__":
    main(sys.argv[1:])
