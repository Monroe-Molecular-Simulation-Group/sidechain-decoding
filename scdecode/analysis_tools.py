"""Functions for performing analysis of backmapping models."""

import sys, os
import glob
import argparse

import numpy as np

import openmm as mm
import parmed as pmd

from . import data_io


def build_bat_histograms(all_bat):
    """
    Creates and collects histograms for each column of a numpy array of BAT coordinates.
    """
    num_atoms = int(all_bat.shape[1] / 3)

    all_hist = {}
    all_edges = {}
    for i in range(all_bat.shape[1]):
        this_hist, this_edges = np.histogram(all_bat[:, i], bins='auto')
        if i < num_atoms:
            label = "bond_%i"%(i + 1)
        elif i >= num_atoms and i < 2*num_atoms:
            label = "angle_%i"%(i - num_atoms + 1)
        elif i >= 2*num_atoms:
            label = "dihedral_%i"%(i - 2*num_atoms + 1)
        all_hist[label] = this_hist
        all_edges["%s_edges"%label] = this_edges

    return all_hist, all_edges


def bat_summary_statistics(files):
    """
    Performs summary statistics over the dataset of BAT coordinates.
    (as a list of .npy files)

    Focuses on building histograms of all BAT degrees of freedom to be predicted.
    """
    all_bat = []
    for f in files:
        all_bat.append(np.load(f)[:, 9:])

    all_bat = np.vstack(all_bat)

    return build_bat_histograms(all_bat)


def compute_bat_stats(arg_list):
    """
    Command line tool for computing BAT distributions.
    """
    parser = argparse.ArgumentParser(prog='analysis_tools.compute_bat_stats',
                                     description='Computes histograms for target BAT coordinates from npy files.',
                                    )
    parser.add_argument('--read_dir', '-r', default='./', help='directory to read files from')
    parser.add_argument('--save_file', '-s', default='bat_stats.npz', help='location/name of file to save to')

    args = parser.parse_args(arg_list)

    files = glob.glob('%s/*.npy'%args.read_dir)

    hist_dict, edges_dict = bat_summary_statistics(files)

    np.savez(args.save_file, **hist_dict, **edges_dict)


# To see which residues having trouble, or if atomic overlaps, helps to have forces
# Have optional flag to return
# That way, can visualize where have high energies/forces by coloring by force if want
def config_energy(coords, sim_obj, compute_forces=False):
    """
    Computes energy of a configuration given a simulation object.
    """
    if not isinstance(coords, mm.unit.Quantity):
        coords = coords * mm.unit.angstrom
    sim_obj.context.setPositions(coords)
    state = sim_obj.context.getState(getEnergy=True, getForces=compute_forces)
    if compute_forces:
        return (state.getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole),
                np.array(state.getForces().value_in_unit(mm.unit.kilojoules_per_mole / mm.unit.angstrom)),
               )
    else:
        return state.getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole)


def sim_from_pdb(pdb_file):
    """
    Creates an OpenMM simulation object from a pdb file.
    Also returns pdb object.
    """
    pdb = mm.app.PDBFile(pdb_file)
    forcefield = data_io.ff
    system = forcefield.createSystem(pdb.topology,
                                     nonbondedMethod=mm.app.NoCutoff)
    integrator = mm.LangevinIntegrator(300*mm.unit.kelvin, 1/mm.unit.picosecond, 0.004*mm.unit.picoseconds)
    simulation = mm.app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    return pdb, simulation


def pdb_energy(pdb_file, compute_forces=False):
    """
    Energy from a pdb file.
    """
    pdb, simulation = sim_from_pdb(pdb_file)
    return config_energy(pdb.positions, simulation, compute_forces=compute_forces)


def pdb_energy_decomp(pdb_file):
    """
    Full decomposition of energies in pdb file using Parmed
    """
    pdb, sim = sim_from_pdb(pdb_file)
    struc = pmd.openmm.load_topology(pdb.topology, system=sim.system, xyz=pdb.positions)
    return pmd.openmm.energy_decomposition_system(struc, sim.system, nrg=mm.unit.kilojoules_per_mole)


def check_cg(xyz_coords, bat_obj):
    """
    Calculates the coarse-grained site position given Cartesian coordinates of the BAT atoms.
    """
    working_ag = bat_obj.atoms.convert_to('PARMED')
    working_ag.coordinates = xyz_coords
    sc_ag = working_ag['!(%s)'%data_io.backbone_atoms]
    masses = np.array([a.mass for a in sc_ag])
    return pmd.geometry.center_of_mass(sc_ag.coordinates, masses)


def check_cg_from_bat(bat_coords, bat_obj):
    """
    Calculates the coarse-grained site position from BAT coordinates.
    """
    xyz_coords = data_io.xyz_from_bat(bat_coords, bat_obj)
    return check_cg(xyz_coords, bat_obj)


if __name__ == "__main__":
    if sys.argv[1] == 'bat_stats':
        compute_bat_stats(sys.argv[2:])
    else:
        print("Argument \'%s\' unrecognized. For the first argument, select \'bat_stats\' or nothing.")
