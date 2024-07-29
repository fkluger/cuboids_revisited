from util.options import *
from util.forward import *
from util.fitting import *
from util.initialisation import *
from util import consistency
from util import em_algorithm
from datasets.nyu_depth.nyu_depth import NYURGBDataset
from datasets.smh import SMHDataset
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.colors import hex2color
rc('text', usetex=False)
plt.rc("font", size=8, family="serif")
import random
import time
import platform
import sklearn.metrics
import pickle
from PIL import Image, ImageFont


def evaluate(opt):
    # tracemalloc.start()

    if opt.visualise:
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 16)
        import open3d as o3d

        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = 'defaultLit'

        renderer = o3d.visualization.rendering.OffscreenRenderer(512, 512)

    total_start = time.time()
    opt = load_opts_for_eval(opt)
    opt.batch = 1
    opt.samplecount = 1
    if opt.seed < 0:
        opt.seed = int(np.random.uniform(0, 100000))

    eval_folder, log = create_eval_folder(opt)

    max_auc_distances = [50, 20, 10, 5]
    auc_values = {}
    best_auc_values = {}
    best_auc_value_idx = {}
    for dist in max_auc_distances:
        auc_values["auc_at_%d" % dist] = []
        best_auc_values["auc_at_%d" % dist] = 0
        best_auc_value_idx["auc_at_%d" % dist] = None
    auc_values["mean"] = []

    hostname = platform.node()
    print("host: ", hostname)
    print("SLURM job ID: ", opt.jobid)

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)

    if opt.dataset == 'nyu':
        valset = NYURGBDataset(data_directory=opt.data_path, split=opt.split, scale=1, split_mat=opt.nyu_split)
        depth_mean = valset.depth_mean
        depth_stdd = valset.depth_stdd
        image_mean = torch.from_numpy(valset.image_mean)
    elif opt.dataset == "smh":
        valset = SMHDataset(opt.data_path, split=opt.split, scale=0.5, keep_in_mem=False)
        depth_mean = valset.depth_mean
        depth_stdd = valset.depth_stdd
        image_mean = torch.from_numpy(valset.image_mean)
    else:
        assert False, "unknown dataset %s" % opt.dataset

    valset_loader = torch.utils.data.DataLoader(valset, shuffle=False, num_workers=0, batch_size=1)

    print_options(opt)

    devices = get_devices(opt)
    fitting_device, consac_device, depth_device, inlier_device = devices
    depth_model = get_depth_model(opt, devices)

    inlier_fun = consistency.soft_inlier_fun_gen(5. / opt.threshold, opt.threshold)
    if not opt.no_occlusion_penalty:
        outlier_fun = consistency.soft_outlier_lin_fun_gen(5. / opt.threshold, opt.threshold, opt.threshold*opt.oai_crossover)
    else:
        outlier_fun = consistency.soft_outlier_fun_gen(5. / opt.threshold, opt.threshold)

    if opt.minsolver == 'lbfgs':
        minimal_solver = CuboidFitLBFGS(a_max=opt.a_max, norm_by_volume=True)
    elif opt.minsolver == 'adam':
        minimal_solver = CuboidFitAdam(a_max=opt.a_max, norm_by_volume=True, lr=0.01 if opt.dataset == "nyu" else 0.5, max_loss=1e-5,
                                       iterations=opt.fitting_iterations, a_min=opt.a_min)
    else:
        minimal_solver = CuboidFitNN(lr=opt.solver_lr, a_max=opt.a_max, a_min=opt.a_min, device=fitting_device,
                                     load=opt.load_solver, layers=4, train=False,
                                     arch=opt.minsolver)

    image_idx = 0

    H, W, Y, H_, W_, Y_, M, P, S, Q, R, B, K, model_dim, data_dim, minimal_set_size, dimensions = get_dimensions(opt, valset)

    consac = get_consac_model(opt, devices)

    if opt.inlier_cutoff < 0:
        opt.inlier_cutoff = 9 * np.log(Y_)

    torch.set_grad_enabled(False)

    lists_of_all_distances = [[] for ki in range(K)]
    lists_of_oa_distances = [[] for ki in range(K)]
    list_of_mean_covered_distances = []
    list_of_mean_coverages = []

    ki = 0

    single_times = []

    print("\n##########\n")

    csv_path = os.path.join(eval_folder, "results.csv")
    csv_file = open(csv_path, "w")
    csv_writer = csv.writer(csv_file, delimiter=" ")
    csv_metrics = ["auc_at_20", "mean"]
    csv_writer.writerow(["idx"] + csv_metrics)

    all_cuboids = []
    all_num_cuboids = []
    all_cuboid_aucs = []
    all_cuboid_inliers = []

    for (image, intrinsic, true_coord_grid, labels, gt_models, gt_depth, file_indices, mask) in valset_loader:

        if image_idx < opt.sampleid and opt.sampleid > -1:
            print("%d.. " % image_idx, end="\r")
            image_idx += 1
            continue

        print("image index: %d / %d " % (image_idx, len(valset_loader)))

        bi = 0

        single_start = time.time()

        target_file = os.path.join(eval_folder, "sample_intermediate_%04d_%04d.pkl" % (image_idx, file_indices[0].item()))


        if opt.load_eval_results is None or not os.path.exists(target_file):

            depth, depth_normalised, depth_mse, images_scaled = \
                estimate_depth(opt, image, image_mean, depth_model, dimensions, devices, depth_mean, depth_stdd, gt_depth,
                               read_cache=opt.read_cache, write_cache=opt.write_cache)

            if opt.write_cache_only:
                image_idx += 1
                continue

            data = depth_normalised.detach().to(consac_device)

            true_coord_grid, true_coord_grid_small, true_coord_flat, \
            estm_coord_grid, estm_coord_grid_small, estm_coord_flat, mask_flat, mask_small = \
                generate_coordinate_grids(depth, true_coord_grid, dimensions, devices, mask)


            states = torch.zeros((P, K, B, H, W, 1), device=consac_device)
            all_probs = torch.zeros((P, M, K, B, Q, R, H_, W_), device=consac_device) #!!
            all_q_probs = torch.zeros((P, M, K, B, Q), device=consac_device) #!!
            all_best_models = torch.zeros((P, M, K, B, model_dim), device=fitting_device) #!!
            all_best_inlier_counts_estm = torch.zeros((P, M, K, B), device=depth_device) #!!
            all_best_inliers_estm_tensor = torch.zeros((P, M, K, B, Y_), device=fitting_device) #!!
            all_best_inliers_gt_tensor = torch.zeros((P, M, K, B, Y_), device=fitting_device) #!!
            all_best_sel_choices = torch.zeros((P, M, K, B, Q), device=fitting_device) #!!

            prev_inliers_estm = None
            prev_inliers_gt = None
            prev_distances_gt = None
            prev_occluded_distances_gt = None

            all_models_per_iter = []
            M_actual = 0

            for mi in range(M):

                print("fitting cuboid %d" % mi, end="\r")

                if prev_inliers_estm is not None:
                    for pi in range(P):
                        inliers_scaled = torch.nn.functional.interpolate(
                            prev_inliers_estm[pi, ki, :].view(B, 1, H_, W_), size=(H, W)).squeeze()
                        states[pi, :, :, :, :, 0] = inliers_scaled

                sampling_weight_maps, selection_weights, log_probs, log_q, entropy = \
                    estimate_sampling_weights(opt, dimensions, devices, data, states, consac, prev_inliers_estm)

                sampling_weight_maps = sampling_weight_maps * mask_flat[None, None, None, :, None, None, :]

                all_probs[:, mi, :] = sampling_weight_maps.view(P, S, K, B, Q, R, H_, W_)[:, 0]
                all_q_probs[:, mi, :] = selection_weights[:, 0]

                models, choices, sel_choices, residual, residuals = \
                    estimate_models(opt, dimensions, estm_coord_flat, sampling_weight_maps[:, :, :, :, :Q].detach(),
                                    selection_weights.detach(), minimal_solver)

                inliers_estm, distances_estm, occluded_distances_estm, oa_distances_estm = \
                    count_inliers(opt, models, estm_coord_flat, inlier_fun, outlier_fun, None, None, prev_inliers_estm,
                                  opt.inlier_mode_selection)
                del distances_estm, occluded_distances_estm

                inliers_estm = inliers_estm * mask_flat[None, None, None, ...]

                best_single_hypos, all_best_models[:, mi] = \
                    select_single_hypotheses(dimensions, inliers_estm, models=models)

                if opt.step_em_iter > 0 and mi > 0:

                    dist_fun = consistency.cuboid_distance_occluded if opt.em_distance == "occluded" else consistency.cuboid_distance_batched

                    em = em_algorithm.CuboidEM(all_best_models[:, :(mi+1)], estm_coord_flat, opt.em_lr, dist_fun,
                                               opt.em_init_variance, False)
                    all_best_models[:, :(mi+1)] = em.run_iterations(opt.step_em_iter)

                all_models_per_iter += [[all_best_models[:, :(mi + 1)].clone().detach().squeeze(2).squeeze(2)]]

                prev_inliers_estm, _, _, _ = \
                    count_inliers(opt, all_best_models[:, mi].unsqueeze(1), estm_coord_flat, inlier_fun, outlier_fun,
                                  None, None, prev_inliers_estm, opt.inlier_mode_selection)

                prev_inliers_estm = prev_inliers_estm * mask_flat[None, None, None, ...]

                prev_inliers_gt, prev_distances_gt, prev_occluded_distances_gt, oa_distances_gt = \
                    count_inliers(opt, all_best_models[:, mi].unsqueeze(1), true_coord_flat, inlier_fun, outlier_fun,
                                  prev_distances_gt, prev_occluded_distances_gt, prev_inliers_gt, opt.inlier_mode_loss)

                prev_inliers_gt = prev_inliers_gt * mask_flat[None, None, None, ...]

                all_best_inliers_estm_tensor[:, mi] = prev_inliers_estm
                best_sel_choices = torch.gather(sel_choices, 1, best_single_hypos.unsqueeze(1).unsqueeze(-1).\
                                                                expand(-1, -1, -1, -1, sel_choices.size(-1))).squeeze(1)
                all_best_sel_choices[:, mi] = best_sel_choices

                all_best_inliers_gt_tensor[:, mi] = prev_inliers_gt

                all_best_inlier_counts_estm[:, mi] = prev_inliers_estm.sum(-1)

                if mi > 0:
                    inlier_increase = all_best_inlier_counts_estm[0, mi, 0, 0] - \
                                      all_best_inlier_counts_estm[0, mi - 1, 0, 0]
                else:
                    inlier_increase = all_best_inlier_counts_estm[0, mi, 0, 0]

                if inlier_increase.item() < opt.inlier_cutoff:
                    break
                else:
                    M_actual += 1

            sample_results = {}
            sample_results["all_probs"] = all_probs.cpu().detach().numpy()
            sample_results["all_q_probs"] = all_q_probs.cpu().detach().numpy()
            sample_results["all_best_inliers_estm_tensor"] = all_best_inliers_estm_tensor.cpu().detach().numpy()
            sample_results["all_best_inliers_gt_tensor"] = all_best_inliers_gt_tensor.cpu().detach().numpy()
            sample_results["depth"] = depth.cpu().detach().numpy()
            sample_results["depth_mse"] = depth_mse.cpu().detach().numpy()
            sample_results["true_coord_grid"] = true_coord_grid.cpu().detach().numpy()
            sample_results["true_coord_flat"] = true_coord_flat.cpu().detach().numpy()
            sample_results["estm_coord_grid"] = estm_coord_grid.cpu().detach().numpy()
            sample_results["estm_coord_flat"] = estm_coord_flat.cpu().detach().numpy()
            sample_results["all_best_models"] = all_best_models.cpu().detach().numpy()
            sample_results["all_best_inlier_counts_estm"] = all_best_inlier_counts_estm.cpu().detach().numpy()
            sample_results["all_best_sel_choices"] = all_best_sel_choices.cpu().detach().numpy()
            sample_results["M_actual"] = M_actual

        else:
            sample_results = pickle.load(open(target_file, "rb"))

            all_probs = torch.from_numpy(sample_results["all_probs"])
            all_q_probs = torch.from_numpy(sample_results["all_q_probs"])
            all_best_inliers_estm_tensor = torch.from_numpy(sample_results["all_best_inliers_estm_tensor"])
            all_best_inliers_gt_tensor = torch.from_numpy(sample_results["all_best_inliers_gt_tensor"])
            depth = torch.from_numpy(sample_results["depth"])
            true_coord_grid = torch.from_numpy(sample_results["true_coord_grid"])
            estm_coord_flat = torch.from_numpy(sample_results["estm_coord_flat"])
            all_best_models = torch.from_numpy(sample_results["all_best_models"])
            all_best_inlier_counts_estm = torch.from_numpy(sample_results["all_best_inlier_counts_estm"])

            M_actual = sample_results["M_actual"]

        if opt.final_em_iter > 0:
            em_start = time.time()
            dist_fun = consistency.cuboid_distance_occluded if opt.em_distance == "occluded" else consistency.cuboid_distance_batched
            em = em_algorithm.CuboidEM(all_best_models, estm_coord_flat, opt.em_lr, dist_fun,
                                       opt.em_init_variance, False)
            models = em.run_iterations(opt.final_em_iter)

            all_best_models = models.view(all_best_models.size())

            print("EM time:  %.2f s" % (time.time()-em_start))

        M_actual = np.maximum(M_actual, 1)

        all_num_cuboids += [M_actual]

        all_num_primitives = torch.ones((P, K, B), device=all_best_inlier_counts_estm.device, dtype=torch.long)
        selected_joint_inlier_counts = all_best_inlier_counts_estm[:, 0]
        for bi in range(B):
            for pi in range(P):
                inlier_increase = all_best_inlier_counts_estm[pi, 0, ki, bi]
                print("cuboid %d : inlier increase: " % 0, inlier_increase.item())
                inliers_total = inlier_increase.item()
                for mi in range(1, M_actual):
                    inlier_increase = all_best_inlier_counts_estm[pi, mi, ki, bi] - \
                                      all_best_inlier_counts_estm[pi, mi-1, ki, bi]
                    print("cuboid %d : inlier increase: " % mi, inlier_increase.item())
                    if inlier_increase.item() < opt.inlier_cutoff:
                        break
                    else:
                        selected_joint_inlier_counts[pi, ki, bi] = all_best_inlier_counts_estm[pi, mi, ki, bi]
                        all_num_primitives[pi, ki, bi] += 1
                    inliers_total += inlier_increase.item()
                print("inliers total: ", inliers_total)



        best_outer_hypos = torch.argmax(selected_joint_inlier_counts, dim=0) # KxB

        best_num_primitives = torch.gather(all_num_primitives, 0, best_outer_hypos.view(1, K, B).expand(1, K, B)).squeeze(0)
        best_models = torch.gather(all_best_models.to(best_outer_hypos.device), 0, best_outer_hypos.view(1, 1, K, B, 1).expand(1, M, K, B, model_dim)).squeeze(0)


        single_time = time.time() - single_start
        print("time elapsed: %.2f s" % single_time)
        single_times += [single_time]

        occluded_distances_gt = None
        distances_gt = None
        inliers_gt = None
        all_oa_distances_gt = []
        all_signed_oa_distances_gt = []
        all_l2_distances_gt = []
        auc_per_cuboid = []
        inliers_per_cuboid = []
        for mi in range(M_actual):
            inliers_gt, distances_gt, occluded_distances_gt, oa_distances_gt = \
                count_inliers(opt, best_models[mi].view(1,1,1,1,9).to(true_coord_grid.device),
                              true_coord_grid.view(B, Y, 3), inlier_fun, outlier_fun,
                              distances_gt, occluded_distances_gt, inliers_gt, "occlusion_aware")
            all_oa_distances_gt += [torch.sqrt(oa_distances_gt).squeeze().cpu().detach().numpy()]
            all_l2_distances_gt += [torch.sqrt(distances_gt).squeeze().cpu().detach().numpy()]

            auc_per_cuboid += [consistency.calc_auc_values_np(all_oa_distances_gt[-1])]

            inlier_count = torch.sum(inliers_gt).detach().cpu().item()
            inliers_per_cuboid += [inlier_count]

            signed_oa_distances = torch.where(occluded_distances_gt > 0, -torch.sqrt(occluded_distances_gt), torch.sqrt(distances_gt))
            all_signed_oa_distances_gt += [signed_oa_distances.squeeze().cpu().detach().numpy()]



        all_cuboids += [best_models.squeeze().detach().cpu().numpy()]
        all_cuboid_aucs += [auc_per_cuboid]
        all_cuboid_inliers += [inliers_per_cuboid]

        distances_gt = torch.sqrt(distances_gt)

        oa_distances_gt = torch.sqrt(oa_distances_gt)

        cuboid_coverage_2d = calc_cuboid_coverage(best_models[:M_actual].view(M_actual, 9), (W, H), intrinsic)

        cuboid_coverage = np.mean(cuboid_coverage_2d)
        print("coverage: ", cuboid_coverage)
        oa_mask = np.logical_and(cuboid_coverage_2d.T.flatten(), mask.flatten().cpu().detach().numpy())
        covered_oa_distances = oa_distances_gt.squeeze().cpu().detach().numpy()[oa_mask.nonzero()]

        if covered_oa_distances.size > 0:
            mean_covered_oa_distance = covered_oa_distances.mean()
            print("mean covered OA: ", mean_covered_oa_distance)
            list_of_mean_covered_distances += [mean_covered_oa_distance]

        list_of_mean_coverages += [cuboid_coverage]

        depth_error = (depth[bi, :, :].cpu().detach().numpy().squeeze() - gt_depth[bi].cpu().detach().numpy().squeeze()) ** 2
        depth_mse = np.mean(depth_error)

        sample_results["image"] = image.cpu().detach().numpy()
        sample_results["depth"] = depth.cpu().detach().numpy()
        sample_results["depth_mse"] = depth_mse
        sample_results["all_best_models"] = all_best_models.cpu().detach().numpy()
        sample_results["all_num_primitives"] = all_num_primitives.cpu().detach().numpy()
        sample_results["best_models"] = best_models.cpu().detach().numpy()
        sample_results["best_num_primitives"] = best_num_primitives.cpu().detach().numpy()
        sample_results["best_outer_hypos"] = best_outer_hypos.cpu().detach().numpy()
        sample_results["all_probs"] = all_probs.cpu().detach().numpy()
        sample_results["all_q_probs"] = all_q_probs.cpu().detach().numpy()
        sample_results["all_best_inliers_estm_tensor"] = all_best_inliers_estm_tensor.cpu().detach().numpy()
        sample_results["all_best_inliers_gt_tensor"] = all_best_inliers_gt_tensor.cpu().detach().numpy()

        if opt.save_all:
            sample_results["distances_gt"] = distances_gt.cpu().detach().numpy()
            sample_results["oa_distances_gt"] = oa_distances_gt.cpu().detach().numpy()

        oa_distances_gt_numpy = oa_distances_gt[ki].cpu().detach().numpy().squeeze()
        oa_distances_gt_numpy = oa_distances_gt_numpy[np.nonzero((mask.flatten().cpu().detach().numpy()))]
        mean_distance = np.mean(oa_distances_gt_numpy)
        print("mean OA distance: %.3f" % mean_distance)
        lists_of_oa_distances[ki] += [mean_distance]
        all_distances_sorted = np.sort(oa_distances_gt_numpy)
        lists_of_all_distances[ki] += all_distances_sorted.tolist()
        inlier_range = np.arange(Y).astype(np.float32) * 1. / Y

        sample_auc_values = {}
        for max_distance in max_auc_distances:
            max_value = max_distance / 100.
            x = np.append(all_distances_sorted[np.where(all_distances_sorted < max_value)], max_value)
            y = inlier_range[np.where(all_distances_sorted < max_value)]
            if y.size > 2:
                y = np.append(y, y[-1])

                auc = sklearn.metrics.auc(x, y) / max_value
            else:
                auc = 0

            sample_auc_values["auc_at_%d" % max_distance] = auc
            auc_values["auc_at_%d" % max_distance] += [auc]

            if auc > best_auc_values["auc_at_%d" % max_distance]:
                best_auc_values["auc_at_%d" % max_distance] = auc
                best_auc_value_idx["auc_at_%d" % max_distance] = {"ki": ki, "idx": image_idx, "fidx": file_indices[0].item()}

        sample_auc_values["mean"] = mean_distance

        auc_values["mean"] += [mean_distance]

        sample_results["sample_%d" % ki] = {"auc": sample_auc_values, "inlier_range": inlier_range,
                                            "all_distances_sorted": all_distances_sorted}

        if ki == 0:
            csv_writer.writerow([image_idx] + [sample_auc_values[x] for x in csv_metrics])

        if not opt.dont_save:
            with open(target_file, "wb") as file:
                pickle.dump(sample_results, file)


        depth_errors = gt_depth[bi].cpu().detach().numpy().squeeze() - depth[bi, :, :].cpu().detach().numpy().squeeze()
        depth_errors *= depth_errors


        gray_image = rgb2gray(image[bi].cpu().detach().numpy().squeeze())

        if opt.visualise:

            import open3d as o3d

            def get_camera(width, height, K):
                focal = (K[0,0]+K[1,1])/2
                intrinsic = o3d.camera.PinholeCameraIntrinsic()
                intrinsic.set_intrinsics(width, height, focal, focal, width / 2.0 - 0.5, height / 2.0 - 0.5)

                extrinsic = np.identity(4, dtype=np.float64)

                cam_params = o3d.camera.PinholeCameraParameters()
                cam_params.extrinsic = extrinsic
                cam_params.intrinsic = intrinsic

                return cam_params

            def get_cuboid_meshes(cuboids):

                M, K, B, _ = cuboids.size()

                colours = ['#e6194b', '#4363d8', '#aaffc3', '#911eb4', '#46f0f0', '#f58231', '#3cb44b', '#f032e6',
                           '#008080', '#bcf60c', '#fabebe', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3',
                           '#e6194b', '#4363d8', '#aaffc3', '#911eb4', '#46f0f0', '#f58231', '#3cb44b', '#f032e6',
                           '#008080', '#bcf60c', '#fabebe', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3'
                           ]

                meshes = []

                for mi in range(M):
                    cuboid = cuboids[mi, ki, 0].squeeze().cpu().numpy().astype(np.float32)

                    colour = np.array(hex2color(colours[mi]))

                    ax = cuboid[0]
                    ay = cuboid[1]
                    az = cuboid[2]
                    q = np.clip(cuboid[3:6], -100, 100)
                    t = np.clip(cuboid[6:9], -100, 100)

                    angle = np.linalg.norm(q, axis=-1, keepdims=True)
                    angle_axis = Quaternion(axis=q / angle, angle=angle.squeeze())
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

            def save_visualised_meshes(meshes, cam_params, W, H):
                vis = o3d.visualization.Visualizer()

                vis.create_window(width=W, height=H, left=0, top=0, visible=False)
                for mesh in meshes:
                    vis.add_geometry(mesh)
                    vis.update_geometry(mesh)
                    vis.get_view_control().convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
                vis.poll_events()
                vis.update_renderer()

                vis.get_render_option()
                vis.get_render_option().point_size = 0.1

                image = vis.capture_screen_float_buffer(True)
                depth = vis.capture_depth_float_buffer(True)
                return image, depth


            cam_params = get_camera(W, H, intrinsic[0].cpu().numpy())

            if True:

                models_ = all_models_per_iter[-1][-1].transpose(0, 1).unsqueeze(1)
                cuboid_meshes = get_cuboid_meshes(models_[:M_actual])

                for i, mesh in enumerate(cuboid_meshes):
                    renderer.scene.add_geometry("cuboid%d" % i, mesh, material)

                renderer.setup_camera(cam_params.intrinsic, np.identity(4, dtype=np.float64))

                depth_image = np.asarray(renderer.render_to_depth_image(z_in_view_space=True))

                true_depth = gt_depth.cpu().numpy().squeeze()
                min_depth = np.min(true_depth)
                max_depth = np.max(true_depth[(true_depth < 999.0).nonzero()])

                fig = plt.figure(figsize=(8, 6))
                plt.imshow(depth_image, vmin=min_depth, vmax=max_depth, cmap='cividis')
                plt.axis('off')
                targetfile = os.path.join(eval_folder, "final_%04d_depth_cuboids.png" % (image_idx))
                fig.savefig(targetfile, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                fig = plt.figure(figsize=(8, 6))
                plt.imshow(true_depth, vmin=min_depth, vmax=max_depth, cmap='cividis')
                plt.axis('off')
                targetfile = os.path.join(eval_folder, "final_%04d_depth_gt.png" % (image_idx))
                fig.savefig(targetfile, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

                for i, mesh in enumerate(cuboid_meshes):
                    renderer.scene.remove_geometry("cuboid%d" % i)

                image_idx += 1

                render_image, render_depth = save_visualised_meshes(cuboid_meshes, cam_params, W, H)
                render_image = Image.fromarray(np.uint8(np.asarray(render_image) * 255.))
                np_render = np.asarray(render_image)

                np_image = np.uint8(image[0].cpu().detach().numpy().squeeze()*255.)

                targetfile = os.path.join(eval_folder, "final_%04d_render_normal.png" % (image_idx))
                img = Image.fromarray(np_render)
                img.save(targetfile)
                targetfile = os.path.join(eval_folder, "final_%04d_image.png" % (image_idx))
                img = Image.fromarray(np_image)
                img.save(targetfile)

                target_file = os.path.join(eval_folder,
                                           "visdata_%04d_%04d.pkl" % (image_idx, file_indices[0].item()))

                visdata = {
                    "points": true_coord_grid.reshape(B, -1, 3)[0].detach().cpu().numpy(),
                    "oa_distances": all_oa_distances_gt[-1],
                    "cuboids": models_.squeeze().cpu().numpy().astype(np.float32)
                }

                pickle.dump(visdata, open(target_file, "wb"))

        image_idx += 1

    csv_file.close()

    metrics = {"hostname": hostname}
    metrics["M_mean"] = float(np.mean(all_num_cuboids))
    metrics["M_std"] = float(np.std(all_num_cuboids))

    print("\nMean time per image: %.2f s \n" % np.mean(single_times))

    metrics["mean_time"] = float(np.mean(single_times))

    print("\nMean coverage: %.3f \n" % np.mean(list_of_mean_coverages))
    print("\nMean covered OA distance: %.3f \n" % np.mean(list_of_mean_covered_distances))
    metrics["coverage"] = float(np.mean(list_of_mean_coverages))
    metrics["covered_oa_distance"] = float(np.mean(list_of_mean_covered_distances))

    print("\nMean Occlusion-Aware Distances:")
    chamfer_mean = np.mean(lists_of_oa_distances[ki])
    chamfer_stdd = np.std(lists_of_oa_distances[ki])
    print("- run %d : %.6f (%.6f)" % (ki, chamfer_mean, chamfer_stdd))
    metrics["oa_%d" % ki] = float(chamfer_mean)

    num_distances = len(lists_of_all_distances[ki])
    all_distances_sorted = np.sort(np.array(lists_of_all_distances[ki]))
    inlier_range = np.arange(num_distances).astype(np.float32) * 1. / num_distances

    if eval_folder is not None:
        target_file = os.path.join(eval_folder, "oa_distances.pkl")
        pickle.dump(all_distances_sorted, open(target_file, "wb"))
        print("saved oa distances: ", target_file)

    if eval_folder is not None:
        with open(os.path.join(eval_folder, 'metrics.txt'), 'w') as f:
            json.dump(metrics, f, indent=2)

    print("\nAUC for Occlusion-Aware Distances:")
    for max_distance in max_auc_distances:
        max_value = max_distance / 100.
        print("- AUC at %d" % max_distance)

        x = np.append(all_distances_sorted[np.where(all_distances_sorted < max_value)], max_value)
        y = inlier_range[np.where(all_distances_sorted < max_value)]
        if y.size > 2:
            y = np.append(y, y[-1])
            auc = sklearn.metrics.auc(x, y) / max_value
        else:
            auc = 0

        print("---- overall AUC: %.6f" % auc)
        metrics["auc_%d_%d" % (max_distance, ki)] = float(auc)


    total_time = time.time() - total_start
    print("\ntime elapsed: %.1f s (%.2f s)" % (total_time, total_time*1./len(valset_loader)))

    all_results = {}

    all_results["auc_values"] = auc_values
    all_results["best_auc_values"] = best_auc_values
    all_results["best_auc_value_idx"] = best_auc_value_idx

    all_results["cuboids"] = all_cuboids
    all_results["cuboid_aucs"] = all_cuboid_aucs
    all_results["cuboid_inliers"] = all_cuboid_inliers
    all_results["num_cuboids"] = all_num_cuboids

    if eval_folder is not None:
        with open(os.path.join(eval_folder, 'metrics.txt'), 'w') as f:
            json.dump(metrics, f, indent=2)

        target_file = os.path.join(eval_folder, "results.pkl")
        pickle.dump(all_results, open(target_file, "wb"))

    return all_results


if __name__ == "__main__":
    opt = get_options()
    evaluate(opt)