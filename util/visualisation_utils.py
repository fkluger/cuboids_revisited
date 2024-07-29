import numpy as np
import pickle

import trimesh
from pyquaternion import Quaternion
import torchgeometry as tgm
import torch

def fexp(x, p):
    return np.sign(x)*(np.abs(x)**p)


def sq_surface(a1, a2, a3, e1, e2, eta, omega):
    x = a1 * fexp(np.cos(eta), e1) * fexp(np.cos(omega), e2)
    y = a2 * fexp(np.cos(eta), e1) * fexp(np.sin(omega), e2)
    z = a3 * fexp(np.sin(eta), e1)
    return x, y, z


def sq_surface_rbm(a1, a2, a3, e1, e2, q, t, eta, omega):
    x = a1 * fexp(np.cos(eta), e1) * fexp(np.cos(omega), e2)
    y = a2 * fexp(np.cos(eta), e1) * fexp(np.sin(omega), e2)
    z = a3 * fexp(np.sin(eta), e1)

    R = tgm.angle_axis_to_rotation_matrix(torch.from_numpy(q).unsqueeze(0)).numpy().squeeze()[:3, :3]

    points = np.stack([x, y, z]).reshape(3, -1)
    points_transformed = np.dot(R.T, points)
    points_transformed[0,:] += t[0]
    points_transformed[1,:] += t[1]
    points_transformed[2,:] += t[2]

    return points_transformed


def sq_normals_rbm(a1, a2, a3, e1, e2, q, eta, omega):

    x = 1/a1 * fexp(np.cos(eta), 2-e1) * fexp(np.cos(omega), 2-e2)
    y = 1/a2 * fexp(np.cos(eta), 2-e1) * fexp(np.sin(omega), 2-e2)
    z = 1/a3 * fexp(np.sin(eta), 2-e1)

    R = tgm.angle_axis_to_rotation_matrix(torch.from_numpy(q).unsqueeze(0)).numpy().squeeze()[:3, :3]

    normals = np.stack([x, y, z]).reshape(3, -1)
    normals = np.dot(R.T, normals)

    return normals


def sq_surface_tapered(a1, a2, a3, e1, e2, eta, omega, Kx, Ky):
    x = a1 * fexp(np.cos(eta), e1) * fexp(np.cos(omega), e2)
    y = a2 * fexp(np.cos(eta), e1) * fexp(np.sin(omega), e2)
    z = a3 * fexp(np.sin(eta), e1)

    fx = Kx * z / a3
    fx += 1
    fy = Ky * z / a3
    fy += 1
    fz = 1

    x = x * fx
    y = y * fy
    z = z * fz

    return x, y, z


def points_on_sq_surface(a1, a2, a3, e1, e2, R, t, Kx, Ky, n_samples=100):
    """Computes a SQ given a set of parameters and saves it into a np array
    """
    assert R.shape == (3, 3)
    assert t.shape == (3, 1)

    eta = np.linspace(-np.pi/2, np.pi/2, n_samples, endpoint=True)
    omega = np.linspace(-np.pi, np.pi, n_samples, endpoint=True)
    eta, omega = np.meshgrid(eta, omega)
    x, y, z = sq_surface(a1, a2, a3, e1, e2, eta, omega)

    # Apply the deformations
    fx = Kx * z / a3
    fx += 1
    fy = Ky * z / a3
    fy += 1
    fz = 1

    x = x * fx
    y = y * fy
    z = z * fz

    # Get an array of size 3x10000 that contains the points of the SQ
    points = np.stack([x, y, z]).reshape(3, -1)
    points_transformed = R.T.dot(points) + t

    x_tr = points_transformed[0].reshape(n_samples, n_samples)
    y_tr = points_transformed[1].reshape(n_samples, n_samples)
    z_tr = points_transformed[2].reshape(n_samples, n_samples)

    return x_tr, y_tr, z_tr, points_transformed


def points_on_cuboid(a1, a2, a3, e1, e2, R, t, n_samples=100):
    """Computes a cube given a set of parameters and saves it into a np array
    """
    assert R.shape == (3, 3)
    assert t.shape == (3, 1)

    X = np.array([
        [0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 1, 1]
    ], dtype=np.float32)
    X[X == 1.0] = a1
    X[X == 0.0] = -a1

    Y = np.array([
        [0, 0, 0, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0]
    ], dtype=np.float32)
    Y[Y == 1.0] = a2
    Y[Y == 0.0] = -a2

    Z = np.array([
        [1, 1, 0, 0, 1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1]
    ], dtype=np.float32)
    Z[Z == 1.0] = a3
    Z[Z == 0.0] = -a3

    points = np.stack([X, Y, Z]).reshape(3, -1)
    points_transformed = R.T.dot(points) + t
    print("R:", R)
    print("t:", t)

    assert points.shape == (3, 18)

    x_tr = points_transformed[0].reshape(2, 9)
    y_tr = points_transformed[1].reshape(2, 9)
    z_tr = points_transformed[2].reshape(2, 9)
    return x_tr, y_tr, z_tr, points_transformed


def _from_primitive_parms_to_mesh(primitive_params):
    if not isinstance(primitive_params, dict):
        raise Exception(
            "Expected dict and got {} as an input"
            .format(type(primitive_params))
        )
    # Extract the parameters of the primitives
    a1, a2, a3 = primitive_params["size"]
    e1, e2 = primitive_params["shape"]
    Kx, Ky = primitive_params["tapering"]
    t = np.array(primitive_params["location"]).reshape(3, 1)
    R = Quaternion(primitive_params["rotation"]).rotation_matrix.reshape(3, 3)

    # Sample points on the surface of its mesh
    _, _, _, V = points_on_sq_surface(a1, a2, a3, e1, e2, R, t, Kx, Ky)
    assert V.shape[0] == 3

    color = np.array(primitive_params["color"])
    color = (color*255).astype(np.uint8)

    # Build a mesh object using the vertices loaded before and get its
    # convex hull
    m = trimesh.Trimesh(vertices=V.T).convex_hull
    # Apply color
    for i in range(len(m.faces)):
        m.visual.face_colors[i] = color

    return m


def save_primitive_as_ply(primitive_params, filepath):
    m = _from_primitive_parms_to_mesh(primitive_params)
    # Make sure that the filepath endswith .obj
    if not filepath.endswith(".ply"):
        raise Exception(
            "The filepath should have an .ply suffix, instead we received {}"
            .format(filepath)
        )
    m.export(filepath, file_type="ply")


def save_prediction_as_ply(primitive_files, filepath):
    if not isinstance(primitive_files, list):
        raise Exception(
            "Expected list and got {} as an input"
            .format(type(primitive_files))
        )
    m = None
    for p in primitive_files:
        # Parse the primitive parameters
        prim_params = pickle.load(open(p, "r"))
        _m = _from_primitive_parms_to_mesh(prim_params)
        m = trimesh.util.concatenate(_m, m)

    m.export(filepath, file_type="ply")