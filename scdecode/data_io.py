"""
Defines routines for generating training inputs and reading that data for training.
"""

import sys, os
import json
import pickle
import argparse
import glob

import numpy as np

import parmed as pmd

from openmm import app as mmapp

import MDAnalysis as mda
from MDAnalysis.analysis.bat import BAT

import tensorflow as tf


# Define some global variables, like the forcefield used
ff = mmapp.ForceField('amber14/protein.ff14SB.xml')
ref_res_types = [key for key in ff._templates]
ref_res_types.sort()
ref_atom_types = [key for key in ff._atomTypes]
ref_atom_types.sort()
all_ref_types = np.array(ref_atom_types + ref_res_types)
# bat_exclude_dofs = [0, 1, 2, 3, 4, 6]
# XYZ of root atom, angles for first bond in spherical coords, first bond length
# Will be same for ANY residue except GLY
# Except easier to use C-CA-CB as root atoms, which means sidechain is just all
# BAT coords except the first 9 (and this can then include the HA position)
backbone_atoms ='@N,CA,C,O,H,HA,OXT,H1,H2,H3'
not_bat_atoms = '@N,O,H,OXT,H1,H2,H3'
cg_atoms = '@N,CA,C,O,H,CB,OXT'


def inputs_from_pdb(pdb_file, res_name, mod_info,
                    bb_atom_str=backbone_atoms,
                    not_bat_atom_str=not_bat_atoms,
                    cg_atom_list=cg_atoms[1:].split(','),
                    positions=None,
                    rng=np.random.default_rng(),
                    ):
    """
    Generates inputs needed for training sidechain backmapping models from pdb files.

    Parameters
    ----------
    pdb_file : str
        Full path file name for the PDB file to process
    res_name : str
        The name of the residue that will be used for training. All such residues in the
        protein are identified with input/target data for training created for each.
    mod_info : str
        A list of residue indices for which mutations and/or modifications, such as
        adding heavy atoms, have been made. These will not be considered for preparing
        training inputs, even if they match res_name.
    bb_atom_str : str, default '@N,CA,C,O,H,HA,OXT,H1,H2,H3'
        A string following the ParmEd atom selection format specifying backbone atoms.
        Includes the C-terminal and N-terminal backbone atoms.
    not_bat_atom_str : str, default '@N,O,H,OXT,H1,H2,H3'
        A string following ParmEd atom selection format specifying which atoms in a residue
        will NOT be included when generating BAT coordinates. Also excludes the special
        atoms of terminal residues. For GLY, instead want '@N,H,OXT,H1,H2,H3'
    cg_atom_list : list, default ['N', 'CA', 'C', 'O', 'H', 'CB', 'OXT']
        A list of atom names that will be included in the "CG" representation of a protein.
    positions : np.array
        Numpy array specifying coordinates of the protein structure. If provided, will ignore
        the coordinates in the actual PDB file.
    rng : object, default np.random.default_rng()
        Random number generator

    Returns
    -------
    cg_inputs : list of NumPy arrays
        A list of the CG reference beads for each residue to be included in the training set.
        The number of arrays in the list will match the number of residues matching res_name
        and not excluded by mod_info. Each array will be of shape (3,), containing the x, y, z
        coordinates of the CG bead to decode.
    coord_inputs : list of NumPy arrays
        A list of protein coordinates to inform the decoding, including both atomistic and CG.
        Will NOT include the coordinates of the atoms that the model will be trained to decode.
        The number of arrays will be the same as in cg_inputs, but the shape of each array
        will be (N_particles, 3), where N_particles may vary.
    one_hot_inputs : list of NumPy arrays
        Matching one-hot atom/CG bead type information to complement coord_inputs. Generally,
        this will be (N_particles, N_types), where each N_particles entry should match that
        in coord_inputs.
    full_bat : NumPy array
        Array of the full BAT representation for the residues to train on. This includes the
        location of the backbone carbon (C), alpha carbon (CA) and beta carbon (CB), which
        are not part of what is decoded. However, these locations are necessary to convert
        back from BAT to XYZ coordinates.
    bat_targets : list of NumPy arrays
        Coordinates, in a BAT reference frame, of the atoms to be decoded. The number of
        arrays in the list will match that in cg_inputs, with the shape of each array
        being (N_bat,), where N_bat will be the same for a specific residue type.
    bat_analysis : MDAnalysis BAT analysis object
        The object used to convert between BAT and XYZ coordinates for the residue type
        that will be trained on.
    """
    # Get pdb id from file
    pdb_id = os.path.split(pdb_file)[-1].split('.pdb')[0]

    # Load structure with openmm first
    mm_pdb = mmapp.PDBFile(pdb_file)

    # Load into ParmEd
    if positions is not None:
        struc = pmd.openmm.load_topology(mm_pdb.topology, xyz=positions)
    else:
        struc = pmd.openmm.load_topology(mm_pdb.topology, xyz=mm_pdb.positions)

    # Separate out sidechain atoms and compute CG bead locations
    sc_struc = struc['!(%s)'%bb_atom_str]
    cg_coords = []
    for i, r in enumerate(sc_struc.residues):
        masses = np.array([a.mass for a in r])
        cg_coords.append(pmd.geometry.center_of_mass(sc_struc[':%i'%(i+1)].coordinates, masses))
    cg_coords = np.array(cg_coords)

    # Follow procedure in openmm.app.ForceField.createSystem() to correctly apply
    # forcefield information to this pdb
    ff_data = ff._SystemData(mm_pdb.topology)
    templates_for_residues = ff._matchAllResiduesToTemplates(ff_data, mm_pdb.topology, dict(), False)
    atom_types = [ff_data.atomType[a] for a in mm_pdb.topology.atoms()]
    res_types = [templates_for_residues[r.index].name for r in mm_pdb.topology.residues()]

    # Identify target residues
    # Type should match and include terminal versions of the residue (NXXX or CXXX)
    # Terminal status only impacts backbone, not sidechain atoms, so can decode both
    # But will still exclude modified residues (heavy atoms added)
    target_res_num = []
    for i, r in enumerate(res_types):
        if (res_name in r) and (i not in mod_info):
            target_res_num.append(str(i + 1))

    # If have no residues of desired type, stop now
    if len(target_res_num) == 0:
        return None

    # Use .view to preserve residue and atom indexing
    target_res = struc.view[':'+','.join(target_res_num)]

    # Will need list of atom indices of sidechain atoms in every residue
    # Makes easier to randomly select residues for exclusion to emulate random backmapping
    res_atom_inds = []
    for res in struc.residues:
        res_atom_inds.append([a.idx for a in res.atoms if a.name not in cg_atom_list])

    # Create one-hot encodings for atoms and residues
    # But using overall set of combined atom and residue types
    one_hot_atoms = np.zeros((len(atom_types), len(all_ref_types)))
    for i, t in enumerate(atom_types):
        if t not in all_ref_types:
            raise ValueError("The atom type %s does not match any type in the force field."%t)
        one_hot_atoms[i, :] = np.array(all_ref_types == t, dtype='float32')

    one_hot_res = np.zeros((len(res_types), len(all_ref_types)))
    for i, t in enumerate(res_types):
        if t not in all_ref_types:
            raise ValueError("The residue type %s does not match any template in the force field."%t)
        one_hot_res[i, :] = np.array(all_ref_types == t, dtype='float32')

    # Now have all information for entire protein
    # Will loop over residues matching target type and select out for each
    cg_inputs = []
    coord_inputs = []
    one_hot_inputs = []
    full_bat = []
    bat_targets = []

    # Also return BAT analysis object from last target residue
    # Should behave the same way for all residues of same type
    bat_analysis = None

    for res in target_res.residues:
    
        # When selecting atoms, only exclude non-CG representation atoms
        # To be consitent with Rosetta, the CG atoms are the backbone atoms plus the beta carbon (and a CG bead)
        # So only want to remove the FG coordinates of other atoms in the residue
        this_atom_inds = [a.idx for a in res.atoms if a.name not in cg_atom_list]

        # To improve ability to decode structures in any order (and save memory)
        # Randomly select a number of residues, then specific residues, to exclude atoms for
        num_res_exclude = rng.integers(0, len(res_types) + 1)
        res_exclude = rng.choice(len(res_types), size=num_res_exclude, replace=False)
        for i in res_exclude:
            if i != res.idx:
                this_atom_inds.extend(res_atom_inds[i])

        # And gather Cartesian coordinate and type information we want
        this_fg_coords = np.delete(struc.coordinates, this_atom_inds, axis=0)
        this_cg_coords = np.delete(cg_coords, res.idx, axis=0)
        this_fg_one_hot = np.delete(one_hot_atoms, this_atom_inds, axis=0)
        this_cg_one_hot = np.delete(one_hot_res, res.idx, axis=0)
        cg_inputs.append(cg_coords[res.idx, :])
        coord_inputs.append(np.vstack([this_fg_coords, this_cg_coords]))
        one_hot_inputs.append(np.vstack([this_fg_one_hot, this_cg_one_hot]))

        # Also need BAT coordinates for just this residue
        this_bat_atoms = struc['(:%i)&(!(%s))'%(res.idx + 1, not_bat_atom_str)]
        uni = mda.Universe(this_bat_atoms.topology, np.array(this_bat_atoms.coordinates, dtype='float32'))
        if res_name == 'GLY':
            bat_analysis = BAT(uni.select_atoms('all'), initial_atom=uni.select_atoms('name O')[0])
        else:
            bat_analysis = BAT(uni.select_atoms('all'), initial_atom=uni.select_atoms('name C')[0])
        bat_analysis.run()
        this_bat = bat_analysis.results.bat[0] # Includes N, CA, and CB atom information
        full_bat.append(this_bat)
        bat_targets.append(this_bat[9:])

    # Clean up and return
    # Only convert full_bat to array - rest better off in list
    # That way easier to combine inputs from multiple pdbs as ragged tensors if want to
    full_bat = np.array(full_bat, dtype='float32')
    # Ensure residue name matches force field
    for a in bat_analysis._ag:
        a.residue.resname = res_name

    return cg_inputs, coord_inputs, one_hot_inputs, full_bat, bat_targets, bat_analysis


def _create_example(refs, coords, infos, targets):
    """
    A function to create a serialized tf.train.Example object describing training data.

    Parameters
    ----------
    refs : tf.Tensor
        A tensor describing the reference CG positions to be decoded.
    coords : tf.Tensor
        All coordinate information to be passed as inputs (excluding the atoms to be decoded).
    infos : tf.Tensor
        Information, like one-hot encoding or parameters, associated with coords.
    targets : tf.Tensor
        The BAT coordinates to be predicted by the model (targets for training).

    Returns
    -------
    serialized string of tf.train.Example object
    """
    input_list = [tf.io.serialize_tensor(t).numpy() for t in [refs, coords, infos]]
    input_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=input_list))
    target_serialized = tf.io.serialize_tensor(targets)
    target_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[target_serialized.numpy()]))
    features = tf.train.Features(feature={'inputs': input_feature, 'targets': target_feature})
    example = tf.train.Example(features=features)
    return example.SerializeToString()


def _tf_serialize_example(inputs, targets):
    """
    Returns a serialized string representing a tf.train.Example object of training inputs and targets.

    This function is intened to be used to map a dataset of inputs and targets to a serialized
    format that can be saved as a TFRecord.

    Parameters
    ----------
    inputs : list of tf.Tensor objects
        A list of tf.Tensor objects containing the reference CG positions to be decoded, all XYZ
        coordinate information provided for decoding, and particle type information, in that order.
    targets : tf.Tensor
        A tensor containing all training targets (the BAT coordinates to predict).

    Returns
    -------
    example_str : tf.Tensor with dtype str
        Serialized string wrapped in a tensor representing a training example.
    """
    example_str = tf.py_function(_create_example, (inputs[0], inputs[1], inputs[2], targets), tf.string)
    return tf.reshape(example_str, ())


def _create_data_from_serialized_strs(input_str, targets_str):
    """
    Parses a single serialized example, recovering tensors from serialized strings.

    Parameters
    ----------
    input_str : tf.Tensor of dtype str
        The serialized string representing the inputs for training (reference CG
        coordinates, all other coordinates, and particle info).
    targets_str : tf.Tensor of dtype str
        The serialized string representing the targets for training (BAT coords).

    Returns
    -------
    refs_read : tf.Tensor
        Reference CG particle coordinates to decode
    coords_read : tf.Tensor
        Coordinates to be used as inputs the decoding (excludes coordinates
        that are being predicted).
    info_read : tf.Tensor
        One-hot or other information concerning particle types for the supplied
        coordinates.
    targets_read : tf.Tensor
        Targets parsed from the serialized string (BAT coordinates to predict).
    """
    np_inputs = input_str.numpy()
    refs_read = tf.io.parse_tensor(np_inputs[0], out_type='float32')
    coords_read = tf.io.parse_tensor(np_inputs[1], out_type='float32')
    info_read = tf.io.parse_tensor(np_inputs[2], out_type='float32')
    targets_read = tf.io.parse_tensor(targets_str[0].numpy(), out_type='float32')
    return refs_read, coords_read, info_read, targets_read


def _tf_parse_example(example_str):
    """
    Takes a single serialized example string and parses the data from it.

    This is intended to be applied with tf.data.Dataset.map to convert all
    serialized string data into training inputs and targets.

    Parameters
    ----------
    example_str : serialized str
        A serialized string representing a tf.train.Example object.

    Returns
    -------
    tuple of tf.Tensors
        The inputs to the decoding model, as a tuple.
    tf.Tensor
        The targets for the decoding model.
    """
    example_read = tf.io.parse_single_example(
                    example_str,
                    features = {
                                'inputs': tf.io.RaggedFeature(dtype=tf.string),
                                'targets': tf.io.RaggedFeature(dtype=tf.string)}
                    )
    # Must specify shapes for ragged tensors
    refs_spec = tf.TensorSpec(shape=(3,), dtype=tf.float32)
    coords_spec = tf.RaggedTensorSpec(shape=(None, 3), dtype=tf.float32, ragged_rank=0)
    info_spec = tf.RaggedTensorSpec(shape=(None, len(all_ref_types)), dtype=tf.float32, ragged_rank=0)
    data = tf.py_function(_create_data_from_serialized_strs,
                          (example_read['inputs'], example_read['targets']),
                          (refs_spec, coords_spec, info_spec, tf.float32),
                         )
    return (data[0], data[1], data[2]), data[3]


def _data_as_tensors(refs, coords, info, targets):
    """
    Takes lists of NumPy arrays and converts them into tensors.

    Parameters
    ----------
    refs : list of NumPy arrays
        The CG reference coordinates to decode.
    coords : list of NumPy arrays
        Atomic and CG coordinates for the decoder to use.
    info : list of NumPy arrays
        A list of one-hot particle types corresponding to coords.
    targets : list of NumPy arrays
        A list of targets (BAT coordinates) for the model to predict.

    Returns
    -------
    refs : tf.Tensor
        Converted refs
    coords : tf.RaggedTensor
        Converted coords, noting that may be ragged with different
        numbers of particles in each entry.
    info : tf.RaggedTensor
        Converted info, matching numbers of particles in coords.
    targets : tf.Tensor
        Converted targets
    """
    refs = tf.convert_to_tensor(np.array(refs, dtype='float32'))
    targets = tf.convert_to_tensor(np.array(targets, dtype='float32'))

    coords = [np.array(c, dtype='float32') for c in coords]
    info = [np.array(i, dtype='float32') for i in info]
    lens = [c.shape[0] for c in coords]

    coords = np.vstack(coords)
    info = np.vstack(info)

    coords = tf.RaggedTensor.from_row_splits(coords, np.hstack([0, np.cumsum(lens)]))
    info = tf.RaggedTensor.from_row_splits(info, np.hstack([0, np.cumsum(lens)]))

    return refs, coords, info, targets


def save_dataset(refs, coords, info, targets, save_file_name):
    """
    Saves a TFRecord-based dataset from tensors of inputs and targets.

    Parameters
    ----------
    refs : tf.Tensor
        The CG reference coordinates to decode.
    coords : tf.RaggedTensor
        Atomic and CG coordinates for the decoder to use.
    info : tf.RaggedTensor
        One-hot particle types corresponding to coords.
    targets : tf.Tensors
        Targets (BAT coordinates) for the model to predict.
    save_file_name : str
        The name of the file to save the dataset to.

    Returns
    -------
    Writes file save_file_name

    """
    inputs_dset = tf.data.Dataset.from_tensor_slices((refs, coords, info))
    targets_dset = tf.data.Dataset.from_tensor_slices(targets)
    full_dset = tf.data.Dataset.zip((inputs_dset, targets_dset))
    serialized_full_dset = full_dset.map(_tf_serialize_example)
    writer = tf.io.TFRecordWriter(save_file_name, options='GZIP')
    for seri_example in serialized_full_dset:
        writer.write(seri_example.numpy())
    writer.close()


def read_dataset(files, include_cg_target=False):
    """
    Reads a TFRecord-based dataset, producing a parsed dataset.

    Parsing is necessary to deserialize the serialized strings representing tf.train.Example objects.

    Parameters
    ----------
    files : list of str
        A list of file names to load data from.
    include_cg_target : bool, default False
        Whether or not to add in the CG reference as a target.
        If this is done, the original targets are ignored and full BAT .npy files are loaded
        as part of the targets.

    Returns
    -------
    parsed_dset : tf.data.Dataset
        A tf dataset of parsed examples.
    """
    raw_dset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    parsed_dset = raw_dset.map(_tf_parse_example)

    if include_cg_target:
        # Need to obtain all full BAT files and load in that data
        # Assumes file naming consistent with list of files (order will matter!)
        # (i.e., name is "something.tfrecord" and "something_full_BAT.npy")
        full_bat_files = [f.split('.tfrecord')[0]+'_full_BAT.npy' for f in files]
        full_bat = np.vstack([np.load(f) for f in full_bat_files])
        full_bat_dset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(full_bat, dtype=tf.float32))
        only_cg_dset = parsed_dset.map(lambda x, y: x[0])
        inputs_dset = parsed_dset.map(lambda x, y: x)
        target_dset = tf.data.Dataset.zip(full_bat_dset, only_cg_dset)
        target_dset = target_dset.map(lambda x, y: tf.ensure_shape(tf.concat([x, y], axis=-1), (full_bat.shape[1] + 3,)))
        shaped_dset = tf.data.Dataset.zip(inputs_dset, target_dset)
    
    else:
        # Also need to ensure shape of target values
        target_shape = next(iter(parsed_dset))[1].shape
        shaped_dset = parsed_dset.map(lambda inputs, targets: (inputs, tf.ensure_shape(targets, target_shape)))

    return shaped_dset


def main(arg_list):
    parser = argparse.ArgumentParser(prog='data_io.py',
                                     description='Generates training inputs for all pdb files in a directory',
                                    )
    parser.add_argument('mod_file', help="json file containing dictionary of residue modifications for each pdb")
    parser.add_argument('res_type', help="residue type to prepare inputs for")
    parser.add_argument('--read_dir', '-r', default='./', help="directory to read files from")
    parser.add_argument('--save_dir', '-s', default='./', help="directory to save outputs to")
    parser.add_argument('--traj', help="trajectory file to take coordinates for SINGLE pdb structure in read_dir")

    args = parser.parse_args(arg_list)

    # Get dictionary of modifications
    with open(args.mod_file, 'r') as f:
        mod_info = json.load(f)

    # Get a list of all pdb files to process
    pdb_list = glob.glob(os.path.join(args.read_dir, '*.pdb'))
    print('Found %i pdb files to process.'%len(pdb_list))

    # Get a fixed seed for this run and report it
    seed = np.random.SeedSequence()
    rng = np.random.default_rng(seed=seed)
    print('\nSeed entropy: %i'%seed.entropy)
    print('For reproducible results, create a seed with np.random.SeedSequence(entropy=%i)'%seed.entropy)
    print('Then pass that seed to np.random.default_rng(seed=seed), replacing the appropriate lines of code.\n')

    if args.res_type == 'GLY':
        this_not_bat = '@N,H,OXT,H1,H2,H3'
    else:
        this_not_bat = not_bat_atoms
           
    saved_bat_obj = False

    # If a trajectory is provided, loop over it assuming single pdb structure used as topology
    # Will create training dataset for just this one structure and its trajectory
    if args.traj is not None:
        pdb_id = os.path.split(pdb_list[0])[-1].split('.pdb')[0]
        this_mod = {pdb_id: []}
        uni = mda.Universe(pdb_list[0], args.traj)
        all_refs = []
        all_coords = []
        all_info = []
        all_bat_targets = []
        all_full_bat = []
        for frame in uni.trajectory:
            inputs = inputs_from_pdb(pdb_list[0], args.res_type, this_mod, rng=rng, not_bat_atom_str=this_not_bat,
                                     positions=frame.positions)
            all_refs.extend(inputs[0])
            all_coords.extend(inputs[1])
            all_info.extend(inputs[2])
            all_bat_targets.extend(inputs[4])
            all_full_bat.append(inputs[3])
            
            if not saved_bat_obj:
                pickle.dump(inputs[-1], open(os.path.join(args.save_dir, '%s_BAT_object.pkl'%(args.res_type)), 'wb'))
                saved_bat_obj = True

        save_dataset(*_data_as_tensors(all_refs, all_coords, all_info, all_bat_targets),
                     os.path.join(args.save_dir, '%s_traj_%s.tfrecord'%(pdb_id, args.res_type))
                    )

        np.save(os.path.join(args.save_dir, '%s_traj_%s_full_BAT.npy'%(pdb_id, args.res_type)),
                np.concatenate(all_full_bat, axis=0),
                allow_pickle=False
               )

    # Otherwise, loop over all pdb files individually
    else:
        for p in pdb_list:
            pdb_id = os.path.split(p)[-1].split('.pdb')[0]
            this_mod = mod_info[pdb_id]
    
            try:
                inputs = inputs_from_pdb(p, args.res_type, this_mod, rng=rng, not_bat_atom_str=this_not_bat)
            except Exception as exc:
                print('On file %s, failed with exception:\n%s'%(p, str(exc)))
                inputs = None
    
            # Check to make sure had residues to work with
            if inputs is not None:
    
                # Want to save as something that tensorflow can read easily and quickly without memory issues
                # Best way is to save as serialized tf.train.Example objects in a tf.data.Dataset in TFRecord file
                refs, coords, info, bat_targets = _data_as_tensors(inputs[0], inputs[1], inputs[2], inputs[4])
                save_dataset(refs, coords, info, bat_targets,
                             os.path.join(args.save_dir, '%s_%s.tfrecord'%(pdb_id, args.res_type))
                            )
    
                # np.savez_compressed(os.path.join(args.save_dir, '%s_%s_inputs.npz'%(pdb_id, args.res_type)),
                #                     cg=inputs[0],
                #                     coords=inputs[1],
                #                     one_hot=inputs[2],
                #                     full_bat=inputs[3],
                #                     bat=inputs[4],
                #                    )
    
                # Specifically save the full set of BAT coordinates
                np.save(os.path.join(args.save_dir, '%s_%s_full_BAT.npy'%(pdb_id, args.res_type)),
                        inputs[3],
                        allow_pickle=False
                       )
    
                # Pickle the most recent BAT object
                # Only need to do once per residue type
                # So do as soon as inputs not None
                if not saved_bat_obj:
                    pickle.dump(inputs[-1], open(os.path.join(args.save_dir, '%s_BAT_object.pkl'%(args.res_type)), 'wb'))
                    saved_bat_obj = True


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    main(sys.argv[1:])
