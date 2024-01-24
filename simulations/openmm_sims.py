import sys, os
import argparse

import numpy as np

import openmm as mm
from openmm import app as mmapp
import parmed as pmd

from scdecode import data_io


def protein_sim(pdb_file, implicit_solvent=True, restrain_cg=False, T=300.0, n_steps=10000):
    """
    Runs simulation in implicit solvent (or vacuum).
    If restrain_cg is True, restrains the CG atoms (mostly backbone).
    Note that will NOT energy minimize before simulating!
    """
    forcefield = data_io.ff
    
    # Get prefix for output files
    out_name = pdb_file.split('.pdb')[0]

    if implicit_solvent:
        print("Using implicit solvent (gbn2 or igb8).")
        forcefield.loadFile('implicit/gbn2.xml') # AMBER igb8

    pdb = mmapp.PDBFile(pdb_file)
    # Should be amber14SB force field
    system = forcefield.createSystem(pdb.topology,
                                     nonbondedMethod=mmapp.NoCutoff,
                                     constraints=mmapp.HBonds,
                                     removeCMMotion=True,
                                     )

    if restrain_cg:
        out_name = out_name + '_restrained'
        # Restrain particles involved in CG representation
        # Choosing to restrain because freezing causes issues with also satisfying H-bond constraints
        # With the spring constant of 200000.0 kJ/mol*nm^2, restraining a hydrogen of mass 1.008 g/mol
        # results in a period of oscillation of 2*pi*sqrt(m/k) ~ 0.014 ps
        # This is a very stiff restraint, limiting motion to around 0.05 Angstroms at 2*kB*T with T=300 K
        # Note use of 0.001 ps timestep, which is smaller than typical for protein simulations
        restraint = mm.CustomExternalForce('k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
        system.addForce(restraint)
        restraint.addGlobalParameter('k', 200000.0*mm.unit.kilojoules_per_mole/(mm.unit.nanometer**2))
        restraint.addPerParticleParameter('x0')
        restraint.addPerParticleParameter('y0')
        restraint.addPerParticleParameter('z0')
        restrain_names = data_io.cg_atoms[1:].split(',')
        for a in pdb.topology.atoms():
            if a.name in restrain_names:
                restraint.addParticle(a.index, pdb.positions[a.index])
        time_step = 0.001 * mm.unit.picosecond
        write_freq = 10000
    else:
        time_step = 0.002 * mm.unit.picosecond
        write_freq = 5000

    # Set up simulation
    integrator = mm.LangevinMiddleIntegrator(T*mm.unit.kelvin,
                                             1.0/mm.unit.picosecond,
                                             time_step,
                                            )
    sim = mmapp.Simulation(pdb.topology, system, integrator)
    sim.context.setPositions(pdb.positions)

    # Assume energy already minimized (at least in vacuum)
    # Proceed to simulation
    # Note that the reporters here will report energies with the restraint energy included
    # We will want to recompute the energies of just saved configurations without a restraint
    # (with an energy decomposition, possibly) after the simulation is done
    sim.reporters.append(mmapp.StateDataReporter('%s.out'%out_name,
                                                 write_freq,
                                                 step=True,
                                                 time=True,
                                                 kineticEnergy=True,
                                                 potentialEnergy=True,
                                                 temperature=True,
                                                 speed=True,
                                                 )
                         )
    sim.reporters.append(pmd.openmm.reporters.NetCDFReporter('%s.nc'%out_name,
                                                             write_freq,
                                                             crds=True,
                                                             frcs=True,
                                                            )
                         )
    sim.step(n_steps)


def main(arg_list):
    parser = argparse.ArgumentParser(prog='openmm_sims',
                                     description='Runs simulation of a protein (implicit solvent)',
                                    )
    parser.add_argument('pdb_file', help='name of pdb file of starting configuration')
    parser.add_argument('--num_steps', '-n', type=int, default=10000, help='number of timesteps for simulation to run')
    parser.add_argument('--no_implicit', action='store_false', help='turns off implicit solvent to just simulate in vacuum') 
    parser.add_argument('--restrain', action='store_true', help='whether or not to restrain the CG atoms (mostly backbone)')

    args = parser.parse_args(arg_list)

    protein_sim(args.pdb_file,
                n_steps=args.num_steps,
                implicit_solvent=args.no_implicit, # Default true, if set, will be false
                restrain_cg=args.restrain, # Default false, so turns on if set
               )


if __name__ == '__main__':
    main(sys.argv[1:])
