"""
Defines functions and a command-line tool for fixing pdb files as inputs for training decoding models of residue sidechains.
"""

import os
import argparse
import json

import numpy as np

import openmm as mm
from pdbfixer import PDBFixer


def prep_pdb(pdbid, missing_cutoff=0.10, mod_cutoff=0.01):
    """
    Fixes/cleans up a provided PDB id using OpenMM's pdbfixer tool.
  
    The resulting clean pdb file is written to "PDBid.pdb", where PDBid is the PDB id
    string that was provided as input.
  
    Parameters
    ----------
    pdbid : str
        The 4-letter PDB id (will download from the RCSB)
    missing_cutoff : float, default=0.1
        A value on the half-open interval (0.0, 1.0] specifying the maximum tolerated
        fraction of missing residues; e.g., if set to 0.2 and 30% of the residues are
        missing, an error will be raised
    mod_cutoff : float, default=0.01
        A value on the half-open interval (0.0, 1.0] specifying the maximum tolerated
        fraction of present residues (not missing) to be modified by adding heavy
        atoms to them. Terminal residues are not considered.
    
    Returns
    -------
    pdb : PDBFixer instance
        A PDBFixer class instance of the cleaned-up PDB; has attributes 'topology' and
        'positions' which can be passed to OpenMM or ParmEd for further processing.
    mod_res_inds : list
        A list of residue indices (global across all chains/molecules) that had atoms
        modified or added.
  
    Notes
    -----
    The file is cleaned up by identifying residues with missing atoms, including hydrogens,
    and filling them in. Residues that are completely absent are not generated and simply
    excluded so that the machine learning model only learns from PDB data, not the output
    of another program. The one exception to this are hydrogen degrees of freedom, which
    may be screened out or handled in some other way during model training. Residues that
    have any heavy atoms added are flagged so that they can be excluded from further
    processing. Again, this is probably a good idea so that the machine learning model is
    only learning from PDB data when generating configurations.
    """

    if (missing_cutoff <= 0.0) or (missing_cutoff > 1.0):
      raise ValueError('Cutoff for maximum tolerated fraction of missing residues is %f '
      'but must be in (0.0, 1.0].'%missing_cutoff)

    pdb = PDBFixer(pdbid=pdbid)

    # Remove ligands, ions, water, etc.
    # Setting to false removes both ligands, ions, AND water
    pdb.removeHeterogens(False)

    # Get total number of residues
    tot_res = sum([len(seq.residues) for seq in pdb.sequences])

    # Find number of missing residues to see if want to use this PDB file
    pdb.findMissingResidues()
    tot_missing = 0
    for key, val in pdb.missingResidues.items():
        tot_missing += len(val)
    if (tot_missing / tot_res) > missing_cutoff:
        raise RuntimeError('The fraction of missing residues is %f, greater than the cutoff '
        'of %f. Will not process this PDB id.'%(tot_missing/tot_res, missing_cutoff))

    # Will not insert any residues, just use data we have
    pdb.missingResidues = {}

    # Check for non-standard residues - raise exception if find any
    # Usually a minor change to the residue, but plenty of configurations it the dataset
    # So can only work with those with standard residues
    pdb.findNonstandardResidues()
    if pdb.nonstandardResidues: 
        raise RuntimeError('Non-standard residues in PDB id %s: %s. Will not process this '
        'PDB id.'%(pdbid, str(pdb.nonstandardResidues)))

    # Find missing atoms, both internal and terminal
    pdb.findMissingAtoms()
    
    # If more than missing_cutoff of residues in sequence are missing heavy atoms, ignore this structure
    # Ignoring terminal residues, which are more likely to have missing atoms
    if (len(pdb.missingAtoms.keys()) / tot_res) > mod_cutoff:
        raise RuntimeError('The fraction of residues missing heavy atoms is %f, greater than the '
        'cutoff of %f. Will not process this PDB id.'%(len(pdb.missingAtoms.keys())/tot_res, mod_cutoff))
    
    # Keep track of added residues
    # Commenting out since will not add anything since not inserting residues
    # res_added_inds = {}
    # for key in pdb.missingResidues.keys():
    #     res_added_inds[key[0]] = []
    # for key, val in pdb.missingResidues.items():
    #     start_ind = len(res_added_inds[key[0]]) + key[1]
    #     res_added_inds[key[0]].extend(list(range(start_ind, start_ind+len(val))))
    
    # Add missing atoms
    pdb.addMissingAtoms()
    
    # Will currently have in terms of residue indices for each chain
    # Want in terms of overall residue indices
    mod_res_inds = []
    # Again commenting because won't have any inserted residues
    # chains = list(pdb.topology.chains())
    # for key, val in res_added_inds.items():
    #     this_res = list(chains[key].residues())
    #     mod_res_inds.extend([this_res[ind].index for ind in val])
    
    # Find residues that had missing atoms added
    # Note that won't work if had residues inserted
    # Non-terminal residues
    for key in pdb.missingAtoms.keys():
        mod_res_inds.append(int(key.index))

    # Terminal residues
    for key in pdb.missingTerminals.keys():
        mod_res_inds.append(int(key.index))

    # Non-standard residues that got replaced
    for tup in pdb.nonstandardResidues:
        mod_res_inds.append(int(tup[0].index))
    
    # Add missing hydrogens
    pdb.addMissingHydrogens(7.0)

    # Check to make sure can apply force field
    ff = mm.app.ForceField('amber14/protein.ff14SB.xml')
    try:
        system = ff.createSystem(pdb.topology)
    except Exception as e:
        raise RuntimeError('Following exception raised when applying force field:\n'
        '%s\n '
        'Will not produce a clean version of this pdb.'%str(e))

    # Write the cleaned up file
    mm.app.PDBFile.writeFile(pdb.topology, pdb.positions, open('%s.pdb'%pdbid, 'w'))

    return pdb, mod_res_inds


def minimize_energy(pdb_file, out_fmt_str='./%s_min.pdb'):
    """
    Energy minimizes a pdb file that is read in and saves a new pdb with the result.

    By default, the original file is not overwritten. The new file will have '_min'
    added to the name, as in original.pdb becomes original_min.pdb.

    Parameters
    ----------
    pdb_file : str
        The pdb file to load.
    out_fmt_str : str, default './%s_min.pdb'
        A format string with %s in it, which will be replaced by the portion of pdb_file
        coming immediately before .pdb

    Returns
    -------
    None - The energy-minimized configuration is written to file
    """
    # Figure out naming of output file
    pdb_id = pdb_file.split('/')[-1].split('.pdb')[0] 
    out_name = out_fmt_str%pdb_id
    if os.path.exists(out_name):
        raise ValueError('Path %s already exists - will not overwrite.'%out_name)

    # Do energy minimization
    pdb = mm.app.PDBFile(pdb_file)
    forcefield = mm.app.ForceField('amber14/protein.ff14SB.xml')
    system = forcefield.createSystem(pdb.topology,
                                     nonbondedMethod=mm.app.NoCutoff)
    integrator = mm.LangevinIntegrator(300*mm.unit.kelvin, 1/mm.unit.picosecond, 0.004*mm.unit.picoseconds)
    simulation = mm.app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy(maxIterations=1000)
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    mm.app.PDBFile.writeFile(pdb.topology, state.getPositions(), open(out_name, 'w'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='prep_pdb.py',
                                     description='Cleans up each PDB in a list of PDB ids.',
                                    )
    parser.add_argument('pdbids', nargs='*')
    parser.add_argument('-f', '--file')

    args = parser.parse_args()

    # If provided file, add to list of PDB ids to process
    all_pdbids = []
    all_pdbids.extend(args.pdbids)
    if args.file is not None:
        with open(args.file) as file:
            f_pdbids = [line.strip() for line in file if not line.strip().startswith('#')]
        all_pdbids.extend(f_pdbids)

    # Loop over all PDB ids
    # Will save dictionary of modified residues for each PDB id
    mod_dict = {}
    for pdbid in all_pdbids:
        try:
            this_pdb, this_mods = prep_pdb(pdbid)
            mod_dict[pdbid] = this_mods
        except Exception as e:
            print("Skipping PDB id %s due to error: \n \t %s"%(pdbid, str(e)))

    # Save dictionary recording modifications
    with open('residue_modifications.json', 'w') as f:
        json.dump(mod_dict, f)
