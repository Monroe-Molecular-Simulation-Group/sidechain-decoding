"""Coordinate transformations."""

import numpy as np
import tensorflow as tf

import MDAnalysis as mda


def bat_cartesian_tf(bat_frame, bat_obj):
    """
    Exactly replicates the Cartesian function already in MDAnalysis.analysis.bat
    
    But it's implemented in tensorflow so that gradients can be computed.
    Do need to provide a bat.BAT object as a reference, though.

    Parameters
    ----------
    bat_frame : NumPy array or tf.Tensor
        The full set of BAT coordinates over a number of frames.
    bat_obj : MDAnalysis BAT analysis object
        The BAT analysis object defining the transformation.

    Returns
    -------
    xyz_coords : tf.Tensor
        The Cartesian coordinates of the residue/sidechain.
    """

    #Want to be able to operate on multiple frames simultaneously
    #(like a batch), so add dimension if just one configuration
    # Actually, just make user do this
    # if len(tf.shape(bat_frame)) == 1:
    #     bat_frame = tf.reshape(bat_frame, (1, -1))
    n_batch = tf.shape(bat_frame)[0]

    # Split the bat vector into more convenient variables
    origin = tf.identity(bat_frame[:, :3])
    (phi, theta, omega) = tf.split(bat_frame[:, 3:6], 3, axis=1)
    (r01, r12, a012) = tf.split(bat_frame[:, 6:9], 3, axis=1)
    n_torsions = (bat_obj._ag.n_atoms - 3)
    bonds = tf.identity(bat_frame[:, 9:n_torsions + 9])
    angles = tf.identity(bat_frame[:, n_torsions + 9:2 * n_torsions + 9])
    torsions = tf.identity(bat_frame[:, 2 * n_torsions + 9:])
    # When appropriate, convert improper to proper torsions
    shift = tf.gather(torsions,
                      tf.tile([bat_obj._primary_torsion_indices], (n_batch, 1)),
                      batch_dims=1)
    unique_primary_torsion_bool = np.zeros(len(bat_obj._primary_torsion_indices), dtype=bool)
    unique_primary_torsion_bool[bat_obj._unique_primary_torsion_indices] = True
    shift = tf.where(unique_primary_torsion_bool, x=tf.zeros_like(shift), y=shift)
    torsions = torsions + shift
    # Wrap torsions to between -np.pi and np.pi
    torsions = ((torsions + np.pi) % (2 * np.pi)) - np.pi

    # Set initial root atom positions based on internal coordinates
    p0 = tf.zeros((n_batch, 3))
    p1 = tf.transpose(tf.scatter_nd([[2]], [tf.reshape(r01, (-1,))], (3, n_batch)))
    p2 = tf.concat([r12 * tf.math.sin(a012),
                    tf.zeros((n_batch, 1)),
                    r01 - r12 * tf.math.cos(a012)], axis=1)

    # Rotate the third atom by the appropriate value
    co = tf.squeeze(tf.math.cos(omega), axis=-1)
    so = tf.squeeze(tf.math.sin(omega), axis=-1)
    # $R_Z(\omega)$
    Romega = tf.transpose(tf.scatter_nd([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]],
                                        [co, -so, so, co, tf.ones(n_batch)],
                                        (3, 3, n_batch)),
                          perm=(2, 0, 1))
    p2 = tf.squeeze(tf.linalg.matmul(Romega, tf.expand_dims(p2, axis=-1)))
    # Rotate the second two atoms to point in the right direction
    cp = tf.squeeze(tf.math.cos(phi), axis=-1)
    sp = tf.squeeze(tf.math.sin(phi), axis=-1)
    ct = tf.squeeze(tf.math.cos(theta), axis=-1)
    st = tf.squeeze(tf.math.sin(theta), axis=-1)
    # $R_Z(\phi) R_Y(\theta)$
    Re = tf.transpose(tf.scatter_nd([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 2]],
                                    [cp*ct, -sp, cp*st, ct*sp, cp, sp*st, -st, ct],
                                    (3, 3, n_batch)),
                      perm=(2, 0, 1))
    p1 = tf.squeeze(tf.linalg.matmul(Re, tf.expand_dims(p1, axis=-1)), axis=-1)
    p2 = tf.squeeze(tf.linalg.matmul(Re, tf.expand_dims(p2, axis=-1)), axis=-1)
    # Translate the first three atoms by the origin
    p0 += origin
    p1 += origin
    p2 += origin

    #With tf, can't change part of a tensor alone, so create list and put together at end
    XYZ = [p0, p1, p2]
    XYZ_order = [bat_obj._root_XYZ_inds[0], bat_obj._root_XYZ_inds[1], bat_obj._root_XYZ_inds[2]]

    # Place the remaining atoms
    for i in range(len(bat_obj._torsion_XYZ_inds)):
        (a0, a1, a2, a3) = bat_obj._torsion_XYZ_inds[i]
        this_r01 = bonds[:, i:i+1]
        this_angle = angles[:, i:i+1]
        this_torsion = torsions[:, i:i+1]

        this_p1 = XYZ[XYZ_order.index(a1)]
        this_p3 = XYZ[XYZ_order.index(a3)]
        this_p2 = XYZ[XYZ_order.index(a2)]

        sn_ang = tf.math.sin(this_angle)
        cs_ang = tf.math.cos(this_angle)
        sn_tor = tf.math.sin(this_torsion)
        cs_tor = tf.math.cos(this_torsion)

        v21 = this_p1 - this_p2
        v21 /= tf.math.sqrt(tf.reduce_sum(v21 * v21, axis=-1, keepdims=True))
        v32 = this_p2 - this_p3
        v32 /= tf.math.sqrt(tf.reduce_sum(v32 * v32, axis=-1, keepdims=True))

        vp = tf.linalg.cross(v32, v21)
        cs = tf.reduce_sum(v21 * v32, axis=-1, keepdims=True)

        sn = tf.math.maximum(tf.math.sqrt(1.0 - cs * cs), 0.0000000001)
        vp = vp / sn
        vu = tf.linalg.cross(vp, v21)

        this_coord = this_p1 + this_r01*(vu*sn_ang*cs_tor + vp*sn_ang*sn_tor - v21*cs_ang)
        XYZ.append(this_coord)
        XYZ_order.append(a0)
    XYZ = tf.dynamic_stitch(XYZ_order, XYZ)
    XYZ = tf.transpose(XYZ, perm=(1, 0, 2))
    return XYZ


def xyz_from_bat_numpy(bat_coords, bat_obj):
    """
    Loops over many BAT coordinates to convert back to Cartesian.

    Parameters
    ----------
    bat_coords : NumPy array
        The full set of BAT coordinates for a specific residue/sidechain.
    bat_obj : MDAnalysis BAT analysis object
        The BAT analysis object that can convert between BAT and Cartesian.

    Returns
    -------
    xyz_coords : NumPy array
        The Cartesian coordinates of the residue/sidechain.
    """
    xyz_coords = []
    for bc in bat_coords:
        xyz_coords.append(bat_obj.Cartesian(bc))
    return np.array(xyz_coords)


def fill_in_bat(partial_bat, root_pos):
    """
    Recreates a full set of BAT coordinates from a partial set and the root atom positions.

    Parameters
    ----------
    partial_bat : NumPy array or tf.Tensor
        The partial set of BAT coordinates, not including the root atom (first 3) coordinates.
    root_pos : NumPy array or tf.Tensor
        A N_frames by 3 by 3 array of the positions of the root atoms. For most residues,
        this will be C, CA, and CB, but may be different for something like GLY.

    Returns
    -------
    full_bat : tf.Tensor
        The full set of BAT coordinates, including information on the CA and CB atom
        locations, which is needed for converting back to XYZ coordinates for all
        atoms in a sidechain.
    """
    if len(tf.shape(root_pos)) == 2:
        root_pos = tf.expand_dims(root_pos, 0)
        do_squeeze = True
    elif len(tf.shape(root_pos)) == 3:
        do_squeeze = False
    else:
        raise ValueError('Positions of root atoms must be N_batchx3x3 or 3x3 (if have no batch dimension).')

    if len(tf.shape(partial_bat)) == 1:
        partial_bat = tf.expand_dims(partial_bat, 0)

    n_batch = tf.shape(root_pos)[0]
    p0 = root_pos[:, 0, :]
    p1 = root_pos[:, 1, :]
    p2 = root_pos[:, 2, :]
    v01 = p1 - p0
    v21 = p1 - p2
    r01 = tf.sqrt(tf.reduce_sum(v01 * v01, axis=-1))
    r12 = tf.sqrt(tf.reduce_sum(v21 * v21, axis=-1))
    a012 = tf.acos(tf.reduce_sum(v01 * v21, axis=-1) / (r01 * r12))
    polar = tf.acos(v01[:, 2] / r01)
    azimuthal = tf.atan2(v01[:, 1], v01[:, 0])
    cp = tf.cos(azimuthal)
    sp = tf.sin(azimuthal)
    ct = tf.cos(polar)
    st = tf.sin(polar)
    Rz = tf.stack([[cp * ct, ct * sp, -st], [-sp, cp, tf.zeros(n_batch)], [cp * st, sp * st, ct]])
    Rz = tf.transpose(Rz, perm=(2, 0, 1))
    pos2 = tf.squeeze(tf.linalg.matmul(Rz, tf.expand_dims(p2 - p1, -1)), axis=-1)
    omega = tf.atan2(pos2[:, 1], pos2[:, 0])
    full_bat = tf.concat([p0, azimuthal[:, None], polar[:, None], omega[:, None],
                          r01[:, None], r12[:, None], a012[:, None], partial_bat], axis=1)
  
    if do_squeeze:
        return tf.squeeze(full_bat)
    else:
        return full_bat
