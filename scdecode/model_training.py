
import glob
import tensorflow as tf

import vaemolsim

from .data_io import read_dataset

def train_model(read_dir='./', save_prefix='./sidechain_decoder')
    files = glob.glob('%s/*.tfrecord'%read_dir)
    dset = read_dataset(files)

    # Should shuffle and batch data set

    model = vaemolsim.models.BackmappingOnly()

    optim = tf.optimizers.Adam()

    model.compile(dset, optim)

    # Any callbacks needed? Shouldn't really need annealing

    history = model.train()

    # Save history

    # Save model


