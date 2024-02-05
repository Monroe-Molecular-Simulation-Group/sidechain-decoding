# Runs MC simulation of CG protein ("centroid" representation) with PyRosetta
# Note that should use environment "protein_env.yaml" or uncomment lines for pyrosetta in "tf_protein_env.yaml"
# But will we need pyrosetta in tandem with tensorflow to do VAE-based MC sims?
# Yes if cannot figure out reweighting...
import sys, os
import argparse

import numpy as np

import pyrosetta
pyrosetta.init()


def cg_protein_sim(pdb_file, n_steps=10000000, write_freq=1000):
    """
    Given a pdb file of fully atomistic coordinates, runs a MCMC simulation of a CG
    representation of the protein. Energies along with CG and deterministically
    backmapped MD snapshots will be output.
    """

    # Get prefix for output files
    out_name = pdb_file.split('.pdb')[0].split('/')[-1]

    # Load in the pose
    pose = pyrosetta.pose_from_pdb(pdb_file)

    # Create a switch to CG representation and apply it
    cg_switch = pyrosetta.SwitchResidueTypeSetMover("centroid")
    cg_switch.apply(pose)

    # And a switching function to recover atomimstic coordinates
    # Only truly need to obtain OXT atom position, which is left out
    # in Rosetta centroid representation
    # But also will be interesting to have deterministically backmapped configs
    fa_switch = pyrosetta.SwitchResidueTypeSetMover("fa_standard")

    # Define the energy function
    cen_sfxn = pyrosetta.create_score_function("score3")

    # Set up our MC simulation, starting with a move map
    # Make it so that all backbone degrees of freedom can be moved
    move_map = pyrosetta.MoveMap()
    move_map.set_bb(True)

    # Define kT to be 1.0 - ok since arbitrary for CG force field
    # Also makes easier to get probability from just the energy
    kT = 1.0

    # Create small perturbations to dihedrals
    # Setting so applies 5 perturbations that are all accepted or rejected
    # And turn off MC acceptance criteria of generated configs based on Ramachandran biasing
    # (so that detailed balance is preserved)
    # To make sure not using Ramachandran energy for acceptance, also set score function
    # And increase maximum angle of perturbation (also sets same regardless of secondary structure)
    small_mover = pyrosetta.rosetta.protocols.simple_moves.SmallMover(move_map, kT, 5)
    small_mover.set_preserve_detailed_balance(True)
    small_mover.scorefxn(cen_sfxn)
    small_mover.angle_max(15.0) # Gets just under 60% acceptance on 1UAO

    # Do some for creation of "shear" moves to perturb dihedrals in correlated fashion
    shear_mover = pyrosetta.rosetta.protocols.simple_moves.ShearMover(move_map, kT, 5)
    shear_mover.set_preserve_detailed_balance(True)
    shear_mover.scorefxn(cen_sfxn)
    shear_mover.angle_max(20.0) # Gets just over 60% acceptance on 1UAO

    # Combine the small and shear movers so randomly pick each 50% of the time
    random_mover = pyrosetta.rosetta.protocols.moves.RandomMover()
    random_mover.add_mover(small_mover, 0.5)
    random_mover.add_mover(shear_mover, 0.5)

    # Create MC acceptance criteria and mover to apply that to generated moves
    mc = pyrosetta.MonteCarlo(pose, cen_sfxn, kT)
    trial_mover = pyrosetta.TrialMover(random_mover, mc)

    # First, run equilibration for 1/10th simulation time
    eq_steps = n_steps // 10
    for i in range(eq_steps):
        trial_mover.apply(pose)

    # Reset MC simulation statistics (except overall acceptance rate)
    mc.reset_counters()

    # Run MC simulation for specified number of steps
    print('Starting simulation...')
    print('\nStep    Energy    Acceptance Rate')
    write_steps = []
    cg_energies = []
    for i in range(1, n_steps+1):
        trial_mover.apply(pose)

        if i % write_freq == 0:
            print('%i    %1.6f    %1.6f'%(i, mc.last_accepted_score(), trial_mover.acceptance_rate()))
            write_steps.append(i)
            cg_energies.append(mc.last_accepted_score())
            pose.dump_pdb('frame_%i_%s.pdb'%(i, out_name))
            fa_switch.apply(pose)
            pose.dump_pdb('aa_frame_%i_%s.pdb'%(i, out_name))
            cg_switch.apply(pose)

    # Print some information about the MC simulation
    mc.show_state()

    # Finish saving energies
    write_steps = np.array(write_steps, dtype=int)
    cg_energies = np.array(cg_energies)
    np.savez('energies_%s.npz'%(out_name), steps=write_steps, energies=cg_energies)


def main(arg_list):
    parser = argparse.ArgumentParser(prog='pyrosetta_sims',
                                     description='Runs CG simulation of a protein',
                                    )
    parser.add_argument('pdb_file', help='name of pdb file of starting atomstic configuration')
    parser.add_argument('--num_steps', '-n', type=int, default=10000, help='number of MC steps for simulation to run')
    parser.add_argument('--write_freq', '-w', type=int, default=1000, help='frequency for writing configurations and energies')

    args = parser.parse_args(arg_list)

    cg_protein_sim(args.pdb_file,
                   n_steps=args.num_steps,
                   write_freq=args.write_freq,
                  )


if __name__ == '__main__':
    main(sys.argv[1:])
