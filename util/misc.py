from util.tee import Tee
import numpy as np
import math
import torch
import os
import json
import torchgeometry as tgm
from datetime import datetime
import hashlib
from pyquaternion import Quaternion
import open3d as o3d


class CosineAnnealingCustom:

    def __init__(self, begin, end, t_max):
        self.T_max = t_max
        self.begin = begin
        self.end = end
        self.inv = begin < end

    def get(self, epoch):
        if not self.inv:
            return self.end + (self.begin - self.end) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        else:
            return self.begin + (self.end - self.begin) * (1 - math.cos(math.pi * epoch / self.T_max)) / 2


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def sample_cuboids_uniformly(params, density):

    B = params.size(0)

    a1 = params[:, 0]
    a2 = params[:, 1]
    a3 = params[:, 2]

    q = torch.clamp(params[:, 3:6], -100, 100)
    t = torch.clamp(params[:, 6:9], -100, 100)

    angle_axis = q
    R = tgm.angle_axis_to_rotation_matrix(angle_axis)
    R = R[:, :3, :3]

    area_1 = a2*a3
    area_2 = a1*a3
    area_3 = a2*a1

    num_points_1 = torch.round(area_1 * density).int()
    num_points_2 = torch.round(area_2 * density).int()
    num_points_3 = torch.round(area_3 * density).int()

    max_num_points = torch.max(torch.max(num_points_1, torch.max(num_points_2, num_points_3)))

    num_points = \
        torch.stack([num_points_1, num_points_1, num_points_2, num_points_2, num_points_3, num_points_3], dim=-1)

    unidist = torch.distributions.uniform.Uniform(-1., 1.)
    points = unidist.sample((B, 6, max_num_points, 3)).to(params.device)

    points[:, :, :, 0] *= a1.view(B, 1, 1)
    points[:, :, :, 1] *= a2.view(B, 1, 1)
    points[:, :, :, 2] *= a3.view(B, 1, 1)

    points[:, 0, :, 0] = a1.view(B, 1)
    points[:, 1, :, 0] = -a1.view(B, 1)
    points[:, 2, :, 1] = a2.view(B, 1)
    points[:, 3, :, 1] = -a2.view(B, 1)
    points[:, 4, :, 2] = a3.view(B, 1)
    points[:, 5, :, 2] = -a3.view(B, 1)

    points_transformed = points @ R.view(B, 1, 3, 3)
    points_transformed += t.view(B, 1, 1, 3)

    return points_transformed, num_points


def create_eval_folder(opt):
    if opt.eval_results is None:
        return None, None
    else:
        date_time = datetime.now().strftime("%Y%m%d_%H-%M-%S")

        if opt.seqransac:
            session_id = "seqransac"
        else:
            substring_start = opt.load.find("session_")
            if substring_start >= 0:
                session_id = opt.load[substring_start:substring_start+11]
            else:
                session_id = date_time

        substring_start = opt.load_depth.find("session_")
        if substring_start >= 0:
            session_id2 = opt.load_depth[substring_start:substring_start+11]
        else:
            session_id2 = "original"

        settings_string = "m%d_p%d_s%d_k%d_ic%d_rs%d" % \
        (opt.instances, opt.outerhyps, opt.hyps, opt.samplecount, opt.inlier_cutoff, opt.seed)

        folder = os.path.join(opt.eval_results, opt.split, opt.depth_model, session_id, session_id2, settings_string)

        os.makedirs(folder, exist_ok=True)

        log_file = os.path.join(folder, "output.log")
        log = Tee(log_file, "w", file_only=False)

        with open(os.path.join(folder, 'commandline_args.txt'), 'w') as f:
            json.dump(opt.__dict__, f, indent=2)

        print("Saving results in: %s " % folder)

        return folder, log


def depth_cache_dir(opt, create=True):
    part1 = os.path.split(os.path.split(opt.load_depth)[0])[1]
    part2 = os.path.split(opt.load_depth)[1]

    directory = os.path.join(opt.depth_cache, part1, part2)
    if create:
        os.makedirs(directory, exist_ok=True)
    return directory


def depth_cache_path(opt, image, **kwargs):
    b = image.copy(order='C').view(np.uint8)
    h = hashlib.sha1(b).hexdigest()

    return os.path.join(depth_cache_dir(opt, **kwargs), h)


def write_cuboid_meshes(cuboids, folder, index):

    M, K, B, _ = cuboids.size()

    for ki in range(K):
        for mi in range(M):
            cuboid = cuboids[mi, ki, 0].squeeze().numpy().astype(np.float32)

            ax = cuboid[0]
            ay = cuboid[1]
            az = cuboid[2]
            q = np.clip(cuboid[3:6], -100, 100)
            t = np.clip(cuboid[6:9], -100, 100)

            angle = np.linalg.norm(q, axis=-1, keepdims=True)
            angle_axis = Quaternion(axis=q/angle, angle=angle.squeeze())
            R = angle_axis.rotation_matrix

            filename = os.path.join(folder, "%04d_%d_cuboid_%d.obj" % (index, ki, mi))

            mesh = o3d.geometry.TriangleMesh.create_box(width=2, height=2, depth=2)
            vertices = np.asarray(mesh.vertices) - 1.
            vertices[:, 0] *= ax
            vertices[:, 1] *= ay
            vertices[:, 2] *= az
            vertices = vertices @ R
            vertices[:, 0] += t[0]
            vertices[:, 1] += t[1]
            vertices[:, 2] += t[2]

            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(filename, mesh, write_ascii=True)
