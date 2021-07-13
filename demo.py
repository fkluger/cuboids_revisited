from util import consistency, initialisation, fitting, forward, options
import numpy as np
import torch
import random
import skimage.transform
import time
import skimage.io
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.colors import hex2color
from pyquaternion import Quaternion
rc('text', usetex=False)
plt.rc("font", size=8, family="serif")

opt = options.get_options()
opt.batch = 1
opt.samplecount = 1
opt.outerhyps = 1
opt.depth_model = "bts"
if opt.load is None:
    opt.load = "./models/consac_weights.net"

if int(opt.gpu) > -1:
    opt.depth_gpu = opt.gpu
    opt.consac_gpu = opt.gpu
    opt.fitting_gpu = opt.gpu
    opt.inlier_gpu = opt.gpu

if opt.seed < 0:
    opt.seed = int(np.random.uniform(0, 100000))

torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)

devices = initialisation.get_devices(opt)
fitting_device, consac_device, depth_device, inlier_device = devices
depth_model = initialisation.get_depth_model(opt, devices)

inlier_fun = consistency.soft_inlier_fun_gen(5. / opt.threshold, opt.threshold)

if opt.lbfgs:
    minimal_solver = fitting.CuboidFitLBFGS(a_max=opt.a_max, norm_by_volume=True)
else:
    minimal_solver = fitting.CuboidFitAdam(a_max=opt.a_max, norm_by_volume=True)


consac = initialisation.get_consac_model(opt, devices)

consistency_fun = consistency.cuboids_consistency_measure_iofun_parallel_batched_torch

image_mean = np.array([0.48936939, 0.42036757, 0.39901595], dtype=np.float32)
depth_mean = 2.7163546
image_stdd = np.sqrt(np.array([0.2884224, 0.29518136, 0.3076983], dtype=np.float32))
depth_stdd = np.sqrt(1.3985424)

K = np.array([[opt.f, 0, opt.cx], [0, opt.f, opt.cy], [0, 0, 1]])
Kinv = np.linalg.inv(K)

single_start = time.time()

image = skimage.io.imread(opt.image_path)
image_size = image.shape[0:2]
image = torch.from_numpy(image).unsqueeze(0)

H, W, Y, H_, W_, Y_, M, _, S, Q, R, _, _, model_dim, data_dim, minimal_set_size, dimensions = \
    initialisation.get_dimensions(opt, image_size=image_size)

coord_grid = np.ones((image_size[0], image_size[1], 3), dtype=np.float32)
x = np.arange(0, image_size[1])
y = np.arange(0, image_size[0])
xv, yv = np.meshgrid(x, y)
coord_grid[:, :, 0] = xv
coord_grid[:, :, 1] = yv
coord_grid_ = np.transpose(Kinv @ np.transpose(coord_grid, (0, 2, 1)), (0, 2, 1))
coord_grid_[:, :, 0] /= coord_grid_[:, :, 2]
coord_grid_[:, :, 1] /= coord_grid_[:, :, 2]
coord_grid_[:, :, 2] /= coord_grid_[:, :, 2]
coord_grid = torch.from_numpy(coord_grid_.astype(np.float32)).unsqueeze(0)

torch.set_grad_enabled(False)

depth, depth_normalised, depth_mse = \
    forward.estimate_depth(opt, image, image_mean, depth_model, dimensions, devices, depth_mean, depth_stdd, None,
                   read_cache=opt.read_cache, write_cache=opt.write_cache)

data = depth_normalised.detach().to(consac_device)

_, _, _, \
estm_coord_grid, estm_coord_grid_small, estm_coord_flat = \
    forward.generate_coordinate_grids(depth, coord_grid, dimensions, devices)

states = torch.zeros((1, 1, 1, H, W, 1), device=consac_device)
all_probs = torch.zeros((1, M, 1, 1, Q, R, H_, W_), device=consac_device)
all_q_probs = torch.zeros((1, M, 1, 1, Q), device=consac_device)
all_best_models = torch.zeros((1, M, 1, 1, model_dim), device=fitting_device)
all_best_inlier_counts_estm = torch.zeros((1, M, 1, 1), device=depth_device)
all_best_inliers_estm_tensor = torch.zeros((1, M, 1, 1, Y_), device=fitting_device)

prev_inliers_estm = None

for mi in range(M):

    print("fitting cuboid %d" % mi, end="\r")

    if prev_inliers_estm is not None:
        inliers_scaled = torch.nn.functional.interpolate(
            prev_inliers_estm[0, 0, :].view(1, 1, H_, W_), size=(H, W)).squeeze()
        states[0, :, :, :, :, 0] = inliers_scaled

    sampling_weight_maps, selection_weights, log_probs, log_q, entropy = \
        forward.estimate_sampling_weights(opt, dimensions, devices, data, states, consac)

    all_probs[:, mi, :] = sampling_weight_maps.view(1, S, 1, 1, Q, R, H_, W_)[:, 0]
    all_q_probs[:, mi, :] = selection_weights[:, 0]

    models, choices, sel_choices, residual = \
        forward.estimate_models(opt, dimensions, estm_coord_flat, sampling_weight_maps[:, :, :, :, :Q].detach(),
                        selection_weights.detach(), minimal_solver)

    inliers_estm, distances_estm, occluded_distances_estm = \
        forward.count_inliers(opt, models, estm_coord_flat, inlier_fun, None, None, prev_inliers_estm,
                      occlusion_aware=(not opt.no_oai_sampling))

    best_single_hypos, best_inliers_estm, all_best_models[:, mi] = \
        forward.select_single_hypotheses(opt, dimensions, inliers_estm, models=models)
    prev_inliers_estm = torch.gather(inliers_estm, 1, best_single_hypos.view(1, 1, 1, 1, 1).expand(1, 1, 1, 1, Y_))

    all_best_inliers_estm_tensor[:, mi] = prev_inliers_estm

    inlier_counts_estm = inliers_estm.sum(-1).to(all_best_inlier_counts_estm.device)

    all_best_inlier_counts_estm[:, mi] = \
        torch.gather(inlier_counts_estm, 1,
                     best_single_hypos.view(1, 1, 1, 1).to(inlier_counts_estm.device)).squeeze(1)

all_num_primitives = torch.ones((1, 1, 1), device=all_best_inlier_counts_estm.device, dtype=torch.long)
selected_joint_inlier_counts = all_best_inlier_counts_estm[:, 0]

print("cuboid 0 : inlier increase: ", all_best_inlier_counts_estm[0, 0, 0, 0].item())
for mi in range(1, M):
    inlier_increase = all_best_inlier_counts_estm[0, mi, 0, 0] - \
                      all_best_inlier_counts_estm[0, mi-1, 0, 0]
    print("cuboid %d : inlier increase: " % mi, inlier_increase.item())
    if inlier_increase.item() < opt.inlier_cutoff:
        break
    else:
        selected_joint_inlier_counts[0, 0, 0] = all_best_inlier_counts_estm[0, mi, 0, 0]
        all_num_primitives[0, 0, 0] += 1

best_outer_hypos = torch.argmax(selected_joint_inlier_counts, dim=0)

best_num_primitives = torch.gather(all_num_primitives, 0, best_outer_hypos.view(1, 1, 1).expand(1, 1, 1)).squeeze(0)
best_models = torch.gather(all_best_models.to(best_outer_hypos.device), 0,
                           best_outer_hypos.view(1, 1, 1, 1, 1).expand(1, M, 1, 1, model_dim)).squeeze(0)

single_time = time.time() - single_start
print("time elapsed: %.2f s" % single_time)

all_probs_np = all_probs[best_outer_hypos].cpu().detach().numpy().squeeze()
all_q_probs_np = all_q_probs[best_outer_hypos].cpu().detach().numpy().squeeze()
all_best_inliers_estm_torch = all_best_inliers_estm_tensor[best_outer_hypos].view(M, 1, H_, W_)
all_best_inliers_estm_torch = torch.nn.functional.interpolate(all_best_inliers_estm_torch, size=(H, W)).squeeze()


def plot_visualisation():
    fig = plt.figure(figsize=(18, 8))
    plt.axis('off')

    Vsp = np.maximum(4, Q+1)
    Hsp = M+2

    ax0a = plt.subplot2grid((Vsp, Hsp), (0, 0), colspan=2, rowspan=2)
    ax0a.axis('off')

    ax0d = plt.subplot2grid((Vsp, Hsp), (2, 0), colspan=2, rowspan=2)
    ax0d.axis('off')
    ax0a.imshow(image[0].cpu().detach().numpy().squeeze())
    ax0a.set_title("input image")
    ax0d.imshow(depth[0, :, :].cpu().detach().numpy().squeeze())
    ax0d.set_title("estimated depth")

    gray_image = forward.rgb2gray(image[0].cpu().detach().numpy().squeeze())
    gray_im_scaled = (gray_image - 0.5) * 0.1

    for mi in range(M):

        probs = all_probs_np[mi, :]
        q_probs = all_q_probs_np[mi, :]

        inliers_estm = all_best_inliers_estm_torch[mi].detach().cpu().numpy()

        for qi in range(Q):
            ax1 = plt.subplot2grid((Vsp, Hsp), (qi, mi+2), colspan=1, rowspan=1)
            ax1.axis('off')
            probs_ = probs[qi]
            probs_ = probs_ / np.max(probs_)
            probs_ = skimage.transform.resize(probs_, gray_im_scaled.shape)

            q = q_probs[qi]
            ax1.imshow(probs_)
            if qi == 0:
                ax1.set_title("cuboid %d \n sample weights \n q%d = %.3f" % (mi, qi, q))
            else:
                ax1.set_title("q%d = %.3f" % (qi, q))

        ax2 =  plt.subplot2grid((Vsp, Hsp), (Q, mi+2), colspan=1, rowspan=1)
        ax2.axis('off')
        ax2.imshow(inliers_estm, vmin=-1, vmax=1)
        ax2.set_title("inliers")

    plt.draw()
    plt.pause(.1)


def get_cuboid_meshes(cuboids):

    M, K, B, _ = cuboids.size()

    colours = ['#e6194b', '#4363d8', '#aaffc3', '#911eb4', '#46f0f0', '#f58231', '#3cb44b', '#f032e6',
               '#008080', '#bcf60c', '#fabebe', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3']

    meshes = []

    for ki in range(K):
        for mi in range(M):
            cuboid = cuboids[mi, ki, 0].squeeze().cpu().numpy().astype(np.float32)

            colour = np.array(hex2color(colours[mi]))

            ax = cuboid[0]
            ay = cuboid[1]
            az = cuboid[2]
            q = np.clip(cuboid[3:6], -100, 100)
            t = np.clip(cuboid[6:9], -100, 100)

            angle = np.linalg.norm(q, axis=-1, keepdims=True)
            angle_axis = Quaternion(axis=q/angle, angle=angle.squeeze())
            R = angle_axis.rotation_matrix

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

            mesh.paint_uniform_color(colour.astype(np.float64))

            meshes += [mesh]

    return meshes


def visualise_meshes(meshes, cam_params):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=image_size[1], height=image_size[0], left=0, top=0, visible=True)
    for mesh in meshes:
        vis.add_geometry(mesh)
    vis.get_view_control().convert_from_pinhole_camera_parameters(cam_params)
    vis.run()
    vis.destroy_window()


meshes = get_cuboid_meshes(best_models)

intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(image_size[1], image_size[0], opt.f, opt.f, image_size[1] / 2.0 - 0.5, image_size[0] / 2.0 - 0.5)

extrinsic = np.identity(4, dtype=np.float64)

cam_params = o3d.camera.PinholeCameraParameters()
cam_params.extrinsic = extrinsic
cam_params.intrinsic = intrinsic

plot_visualisation()
visualise_meshes(meshes, cam_params)
