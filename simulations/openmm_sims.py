# For these simulations, use environment file openmm_sim.yaml
# This is mainly to obtain openmmtools capabilities for replica exchange
import sys, os
import argparse

import numpy as np

from netCDF4 import Dataset

import openmm as mm
from openmm import app as mmapp
from openmmtools import states, mcmc, multistate
import parmed as pmd
import MDAnalysis as mda

# from scdecode import data_io

# Define CG atoms
cg_atoms = '@N,CA,C,O,H,CB,OXT'


def convert_tremd_traj(pdb_file, checkpoint_traj_file, replica_info_file, state0_T=300.0*mm.unit.kelvin, out_dir='./'):
    """
    Converts the openmmtools REMD output checkpoint trajectory file to a standard netcdf trajecctory.
    Only pulls out the lowest-temperature replica trajectory (state 0) by deconvoluting state and
    replica indices from the replica_info_file. Since already doing this, also prints energies
    associated with each configuration at the lowest-temperature replica (state 0) in kJ/mol.
    """
    out_name = checkpoint_traj_file.split('.nc')[0].split('/')[-1].split('checkpoint_traj_')[-1]

    # Read in replia information and checkpoint trajectory information
    # Note will exclude first iteration (starting configuration)
    rep_info_dat = Dataset(replica_info_file, 'r')
    ckpt_dat = Dataset(checkpoint_traj_file, 'r')
    ckpt_freq = rep_info_dat.CheckpointInterval
    states = rep_info_dat['states'][1::ckpt_freq]
    energies = rep_info_dat['energies'][1::ckpt_freq]
    positions = ckpt_dat['positions'][1::]

    # Identify which replica is in state 0 and pull out coordinates and energies from that replica
    # At end, should go from (N_frames, N_replicas, ...) to just (N_frames, ...)
    # Note multiplication by kB*T for energies and 10.0 for nm to Angstroms for positions
    # For energies, only taking energy at zeroth state
    where_0 = (states == 0)
    energies = energies[where_0, ...] * mm.unit.MOLAR_GAS_CONSTANT_R * state0_T
    energies = energies[:, 0].value_in_unit(mm.unit.kilojoules_per_mole)
    positions = positions[where_0, ...] * 10.0

    # Close datasets
    rep_info_dat.close()
    ckpt_dat.close()

    # Save energies
    np.savez('%s/energies_%s_tremd.npz'%(out_dir, out_name), energies=energies)

    # Create a universe with the pdb as a topology and the trajectory from the positions in memory
    uni = mda.Universe(pdb_file, positions, format=mda.coordinates.memory.MemoryReader)

    # Write out the new trajectory
    uni.select_atoms("all").write("%s/%s_tremd.nc"%(out_dir, out_name), frames="all")


def protein_sim(pdb_file,
                implicit_solvent=True,
                restrain_cg=False,
                tremd=False,
                T=300.0,
                n_steps=10000000,
                out_dir='./',
                flex_res=None,
                ):
    """
    Runs simulation in implicit solvent (or vacuum).
    If restrain_cg is True, restrains the CG atoms (mostly backbone).
    Note that will NOT energy minimize before simulating!
    If tremd is true, temperature replica exchange MD will be performed with openmmtools.
    Setting flex_res to a list of integers allows only those residues to have unrestrained sidechains
    (all other residue sidechains will be restrained if restrain_cg is True)
    If restain_cg is False, flex_res has no effect
    """
    forcefield = mmapp.ForceField('amber14/protein.ff14SB.xml')
    
    # Get prefix for output files
    out_name = pdb_file.split('.pdb')[0].split('/')[-1]

    if implicit_solvent:
        print("\nUsing implicit solvent (gbn2 or igb8).")
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
        restrain_names = cg_atoms[1:].split(',')
        if flex_res is None:
            flex_res = np.arange(pdb.topology.getNumResidues())
        print("Only residues %s will be flexible, others will have sidechains restrained."%str(flex_res))
        sc_restraint = mm.CustomCentroidBondForce(1, 'k*((x1-x0)^2+(y1-y0)^2+(z1-z0)^2)')
        system.addForce(sc_restraint)
        sc_restraint.addGlobalParameter('k', 200000.0*mm.unit.kilojoules_per_mole/(mm.unit.nanometer**2))
        sc_restraint.addPerBondParameter('x0')
        sc_restraint.addPerBondParameter('y0')
        sc_restraint.addPerBondParameter('z0')
        for r in pdb.topology.residues():
            atom_ids = []
            this_sum_pos_mass = np.zeros(3) * mm.unit.nanometers * mm.unit.amu
            this_sum_masses = 0.0 * mm.unit.amu
            for a in r.atoms():
                if a.name in restrain_names:
                    restraint.addParticle(a.index, pdb.positions[a.index])
                    if a.name == 'CB':
                        atom_ids.append(a.index)
                        this_sum_pos_mass += pdb.positions[a.index] * system.getParticleMass(a.index)
                        this_sum_masses += system.getParticleMass(a.index)
                elif r.index not in flex_res:
                    restraint.addParticle(a.index, pdb.positions[a.index])
                else:
                    atom_ids.append(a.index)
                    this_sum_pos_mass += pdb.positions[a.index] * system.getParticleMass(a.index)
                    this_sum_masses += system.getParticleMass(a.index)
            if r.index in flex_res:
                this_com = this_sum_pos_mass / this_sum_masses
                sc_restraint.addGroup(atom_ids)
                sc_restraint.addBond([0], this_com)
        time_step = 0.001 * mm.unit.picosecond
        write_freq = 10000
    else:
        time_step = 0.002 * mm.unit.picosecond
        write_freq = 5000

    if tremd:
        print("Setting up TREMD")
        # Set up reference thermodynamic state
        ref_state = states.ThermodynamicState(system=system,
                                              temperature=T*mm.unit.kelvin,
                                              )

        # Set up MC move set for between replica exchange attempts
        n_steps_per_swap = write_freq // 10
        mc_moves = mcmc.LangevinDynamicsMove(timestep=time_step,
                                             collision_rate=1.0/mm.unit.picosecond,
                                             n_steps=n_steps_per_swap,
                                             reassign_velocities=False, # Setting True leads to constraint warnings
                                             constraint_tolerance=1e-06,
                                            )

        # If already have started run, check and load, otherwise set up
        # Check here before file gets created by reporter
        if os.path.exists('%s/replica_info_%s.nc'%(out_dir, out_name)):
            do_restart = True
        else:
            do_restart = False

        # Add reporter object to manage output
        reporter = multistate.MultiStateReporter('%s/replica_info_%s.nc'%(out_dir, out_name),
                                                 checkpoint_interval=10,
                                                 checkpoint_storage='%s/checkpoint_traj_%s.nc'%(out_dir, out_name),
                                                )

        if do_restart:
            sim = multistate.ParallelTemperingSampler.from_storage(reporter)

        else:
            # Set up replica exchange simulation
            sim = multistate.ParallelTemperingSampler(mcmc_moves=mc_moves,
                                                      number_of_iterations=n_steps//n_steps_per_swap,
                                                      online_analysis_interval=None,
                                                     )
       
            # Create simulation, equilibrate, then run
            sim.create(ref_state,
                       states.SamplerState(pdb.positions),
                       reporter,
                       min_temperature=T*mm.unit.kelvin,
                       max_temperature=450.0*mm.unit.kelvin,
                       n_temperatures=6,
                      )
            print('\n Running replicas at temperatures: %s\n'%str([s.temperature for s in sim._thermodynamic_states]))
            equil_time = 1.0 * mm.unit.nanosecond
            equil_steps = int(equil_time / time_step)
            sim.equilibrate(equil_steps // n_steps_per_swap)

        sim.run()

        # Save a trajectory of ONLY the lowest temperature replica in standard netcdf trajectory format
        convert_tremd_traj(pdb_file,
                           '%s/checkpoint_traj_%s.nc'%(out_dir, out_name),
                           '%s/replica_info_%s.nc'%(out_dir, out_name),
                           state0_T=T*mm.unit.kelvin,
                           out_dir=out_dir
                          )

    else:
        # Set up simulation
        integrator = mm.LangevinMiddleIntegrator(T*mm.unit.kelvin,
                                                 1.0/mm.unit.picosecond,
                                                 time_step,
                                                )
        integrator.setConstraintTolerance(1e-06)
        sim = mmapp.Simulation(pdb.topology, system, integrator)
        sim.context.setPositions(pdb.positions)

        equil_time = 1.0 * mm.unit.nanosecond

        # Assume energy already minimized (at least in vacuum)
        # Proceed to simulation
        # Run equilibration (without reporting) for 1 ns
        equil_steps = int(equil_time / time_step)
        print("\nEquilibrating for %s or %i steps."%(str(equil_time), equil_steps))
        sim.step(equil_steps)
        print("Done with equilibration.\n\n")

        # Add reporters and run production
        # Note that the reporters here will report energies with the restraint energy included
        # We will want to recompute the energies of just saved configurations without a restraint
        # (with an energy decomposition, possibly) after the simulation is done
        sim.reporters.append(mmapp.StateDataReporter('%s/%s.txt'%(out_dir, out_name),
                                                     write_freq,
                                                     step=True,
                                                     time=True,
                                                     kineticEnergy=True,
                                                     potentialEnergy=True,
                                                     temperature=True,
                                                     speed=True,
                                                     )
                            )
        sim.reporters.append(pmd.openmm.reporters.NetCDFReporter('%s/%s.nc'%(out_dir, out_name),
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
    parser.add_argument('--num_steps', '-n', type=int, default=10000000, help='number of timesteps for simulation to run')
    parser.add_argument('--no_implicit', action='store_false', help='turns off implicit solvent to just simulate in vacuum') 
    parser.add_argument('--restrain', action='store_true', help='whether or not to restrain the CG atoms (mostly backbone)')
    parser.add_argument('--tremd', action='store_true', help='whether or not to perform temperature replica exchange')
    parser.add_argument('--output_dir', '-o', default='./', help='directory where outputs will be written to')
    parser.add_argument('--flex_res', type=int, nargs='*', default=None, help='residue number(s) to keep flexible')

    args = parser.parse_args(arg_list)
    if args.flex_res is not None:
        flex_res = [ri - 1 for ri in args.flex_res]
    else:
        flex_res = args.flex_res

    protein_sim(args.pdb_file,
                n_steps=args.num_steps,
                implicit_solvent=args.no_implicit, # Default true; if set, will be false
                restrain_cg=args.restrain, # Default false, so turns on if set
                tremd=args.tremd, # Default false, so turns on if set
                out_dir=args.output_dir,
                flex_res=flex_res,
               )


if __name__ == '__main__':
    main(sys.argv[1:])
