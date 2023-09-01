"""Functions for performing analysis of backmapping models."""

import glob, pickle

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


def config_energy(coords, sim_obj):
    """
    Computes energy of a configuration given a simulation object.
    """
    if not isinstance(coords, mm.unit.Quantity):
        coords = coords * mm.unit.angstrom
    sim_obj.context.setPositions(coords)
    state = sim_obj.context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole)


def pdb_energy(pdb_file):
    """
    Energy from a pdb file.
    """
    pdb = mm.app.PDBFile(pdb_file)
    forcefield = mm.app.ForceField('amber14/protein.ff14SB.xml')
    system = forcefield.createSystem(pdb.topology,
                                     nonbondedMethod=mm.app.NoCutoff)
    integrator = mm.LangevinIntegrator(300*mm.unit.kelvin, 1/mm.unit.picosecond, 0.004*mm.unit.picoseconds)
    simulation = mm.app.Simulation(pdb.topology, system, integrator)
    return config_energy(pdb.positions, simulation)


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
