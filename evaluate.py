from util.options import *
from util.forward import *
from util.fitting import *
from util.initialisation import *
from util import consistency
from datasets.nyu_depth.nyu_depth import NYURGBDataset
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=False)
plt.rc("font", size=8, family="serif")
import random
import skimage.transform
import time
import platform
import sklearn.metrics
import pickle

total_start = time.time()

opt = get_options()
opt = load_opts_for_eval(opt)
opt.batch = 1
opt.samplecount = 1
if opt.seed < 0:
    opt.seed = int(np.random.uniform(0, 100000))

eval_folder, log = create_eval_folder(opt)

max_auc_distances = [200, 100, 50, 20, 10, 5]
auc_values = {}
best_auc_values = {}
best_auc_value_idx = {}
for dist in max_auc_distances:
    auc_values["auc_at_%d" % dist] = []
    best_auc_values["auc_at_%d" % dist] = 0
    best_auc_value_idx["auc_at_%d" % dist] = None
auc_values["mean"] = []
auc_values["mean_chamfer"] = []

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
    coord_mean = torch.Tensor([-0.03198672, -0.20830469, 2.7163548])
    image_mean = torch.from_numpy(valset.image_mean)
else:
    assert False, "unknown dataset %s" % opt.dataset

valset_loader = torch.utils.data.DataLoader(valset, shuffle=False, num_workers=6, batch_size=1)

print_options(opt)

devices = get_devices(opt)
fitting_device, consac_device, depth_device, inlier_device = devices
depth_model = get_depth_model(opt, devices)

feed_height = depth_model["height"]
feed_width = depth_model["width"]

inlier_fun = consistency.soft_inlier_fun_gen(5. / opt.threshold, opt.threshold)

if opt.lbfgs:
    minimal_solver = CuboidFitLBFGS(a_max=opt.a_max, norm_by_volume=True)
else:
    minimal_solver = CuboidFitAdam(a_max=opt.a_max, norm_by_volume=True)

all_losses = []

image_idx = 0

all_miss_rates = []
miss_rates_per_image = []

H, W, Y, H_, W_, Y_, M, P, S, Q, R, B, K, model_dim, data_dim, minimal_set_size, dimensions = get_dimensions(opt, valset)

consac = get_consac_model(opt, devices)

consistency_fun = consistency.cuboids_consistency_measure_iofun_parallel_batched_torch

torch.set_grad_enabled(False)

all_chamfer_distances = []
lists_of_all_distances = [[] for ki in range(K)]
lists_of_chamfer_distances = [[] for ki in range(K)]
lists_of_oa_distances = [[] for ki in range(K)]
list_of_mean_weighted_mses = []

print("\n##########\n")

for (image, intrinsic, true_coord_grid, labels, gt_models, gt_depth, file_indices) in valset_loader:

    if image_idx < opt.sampleid:
        print("%d.. " % image_idx, end="\r")
        image_idx += 1
        continue

    print("image index: %d / %d " % (image_idx, len(valset_loader)))

    sample_results = {}

    bi = 0

    depth, depth_normalised, depth_mse = \
        estimate_depth(opt, image, image_mean, depth_model, dimensions, devices, depth_mean, depth_stdd, gt_depth,
                       read_cache=opt.read_cache, write_cache=opt.write_cache)

    if opt.write_cache_only:
        image_idx += 1
        continue

    data = depth_normalised.detach().to(consac_device)

    true_coord_grid, true_coord_grid_small, true_coord_flat, \
    estm_coord_grid, estm_coord_grid_small, estm_coord_flat = \
        generate_coordinate_grids(depth, true_coord_grid, dimensions, devices)

    single_start = time.time()

    states = torch.zeros((P, K, B, H, W, 1), device=consac_device)
    all_best_inliers_estm = [torch.zeros((P, K, B, Y_), device=fitting_device) for _ in range(M)]
    all_best_inliers_gt = [torch.zeros((P, K, B, Y_), device=depth_device) for _ in range(M)]
    all_probs = torch.zeros((P, M, K, B, Q, R, H_, W_), device=consac_device)
    all_q_probs = torch.zeros((P, M, K, B, Q), device=consac_device)
    all_best_models = torch.zeros((P, M, K, B, model_dim), device=fitting_device)
    all_best_inlier_counts_estm = torch.zeros((P, M, K, B), device=depth_device)
    all_inlier_counts_estm = torch.zeros((P, M, S, K, B), device=depth_device)
    all_inlier_counts_gt = torch.zeros((P, M, S, K, B), device=depth_device)
    mean_oa_distances_gt = torch.zeros((P, M, S, K, B), device=depth_device)
    all_best_inliers_estm_tensor = torch.zeros((P, M, K, B, Y_), device=fitting_device)
    all_best_inliers_gt_tensor = torch.zeros((P, M, K, B, Y_), device=fitting_device)

    max_distance_to_occlusions_gt = None
    max_distance_to_occlusions_estm = None

    prev_inliers_estm = None
    prev_inliers_gt = None
    prev_distances_gt = None
    prev_occluded_distances_gt = None

    for mi in range(M):

        print("fitting cuboid %d" % mi, end="\r")

        if prev_inliers_estm is not None:
            for pi in range(P):
                for ki in range(K):
                    inliers_scaled = torch.nn.functional.interpolate(
                        prev_inliers_estm[pi, ki, :].view(B, 1, H_, W_), size=(H, W)).squeeze()
                    states[pi, :, :, :, :, 0] = inliers_scaled

        sampling_weight_maps, selection_weights, log_probs, log_q, entropy = \
            estimate_sampling_weights(opt, dimensions, devices, data, states, consac)

        all_probs[:, mi, :] = sampling_weight_maps.view(P, S, K, B, Q, R, H_, W_)[:, 0]
        all_q_probs[:, mi, :] = selection_weights[:, 0]

        models, choices, sel_choices, residual = \
            estimate_models(opt, dimensions, estm_coord_flat, sampling_weight_maps[:, :, :, :, :Q].detach(),
                            selection_weights.detach(), minimal_solver)

        inliers_estm, distances_estm, occluded_distances_estm = \
            count_inliers(opt, models, estm_coord_flat, inlier_fun, None, None, prev_inliers_estm,
                          occlusion_aware=(not opt.no_oai_sampling))
        del distances_estm, occluded_distances_estm

        inliers_gt, distances_gt, occluded_distances_gt = \
            count_inliers(opt, models, true_coord_flat, inlier_fun, prev_distances_gt,
                          prev_occluded_distances_gt, prev_inliers_gt, occlusion_aware=(not opt.no_oai_loss))

        best_single_hypos, best_inliers_estm, all_best_models[:, mi] = \
            select_single_hypotheses(opt, dimensions, inliers_estm, models=models)
        prev_inliers_estm = torch.gather(inliers_estm, 1, best_single_hypos.view(P, 1, K, B, 1).expand(P, 1, K, B, Y_))

        all_best_inliers_estm_tensor[:, mi] = prev_inliers_estm

        prev_inliers_gt = torch.gather(inliers_gt, 1, best_single_hypos.view(P, 1, K, B, 1).expand(P, 1, K, B, Y_))
        prev_distances_gt = torch.gather(distances_gt, 1, best_single_hypos.view(P, 1, K, B, 1).expand(P, 1, K, B, Y_))
        prev_occluded_distances_gt = torch.gather(occluded_distances_gt, 1,
                                                  best_single_hypos.view(P, 1, K, B, 1).expand(P, 1, K, B, Y_))

        oa_distances_gt = torch.max(distances_gt, occluded_distances_gt)

        all_best_inliers_gt_tensor[:, mi] = prev_inliers_gt

        all_inlier_counts_estm[:, mi] = inliers_estm.sum(-1).to(all_inlier_counts_estm.device)
        all_inlier_counts_gt[:, mi] = inliers_gt.sum(-1).to(all_inlier_counts_gt.device)
        mean_oa_distances_gt[:, mi] = oa_distances_gt.to(mean_oa_distances_gt.device).mean(dim=-1)

        all_best_oa_distances_gt = \
            torch.gather(oa_distances_gt, 1,
                         best_single_hypos.view(P, 1, K, B, 1).expand(P, 1, K, B, Y_).to(
                             oa_distances_gt.device)).squeeze(1)

        all_best_inlier_counts_estm[:, mi] = \
            torch.gather(all_inlier_counts_estm[:, mi], 1,
                         best_single_hypos.view(P, 1, K, B).to(all_inlier_counts_estm.device)).squeeze(1)

    all_num_primitives = torch.ones((P, K, B), device=all_best_inlier_counts_estm.device, dtype=torch.long)
    selected_joint_inlier_counts = all_best_inlier_counts_estm[:, 0]
    for bi in range(B):
        for pi in range(P):
            for ki in range(K):
                for mi in range(1, M):
                    inlier_increase = all_best_inlier_counts_estm[pi, mi, ki, bi] - \
                                      all_best_inlier_counts_estm[pi, mi-1, ki, bi]
                    print("cuboid %d : inlier increase: " % mi, inlier_increase.item())
                    if inlier_increase.item() < opt.inlier_cutoff:
                        break
                    else:
                        selected_joint_inlier_counts[pi, ki, bi] = all_best_inlier_counts_estm[pi, mi, ki, bi]
                        all_num_primitives[pi, ki, bi] += 1

    best_outer_hypos = torch.argmax(selected_joint_inlier_counts, dim=0) # KxB

    best_num_primitives = torch.gather(all_num_primitives, 0, best_outer_hypos.view(1, K, B).expand(1, K, B)).squeeze(0)
    best_models = torch.gather(all_best_models.to(best_outer_hypos.device), 0, best_outer_hypos.view(1, 1, K, B, 1).expand(1, M, K, B, model_dim)).squeeze(0)

    distances_gt = \
        consistency.cuboids_consistency_measure_iofun_parallel_batched_torch(
            best_models.to(true_coord_grid.device).unsqueeze(0), true_coord_grid.view(B, Y, 3)).squeeze(0)
    distances_gt = torch.sqrt(distances_gt)

    min_distances_gt, _ = torch.min(distances_gt, dim=0)
    for ki in range(K):
        for bi in range(B):
            min_distances_gt[ki, bi], _ = torch.min(distances_gt[:best_num_primitives[ki, bi], ki, bi], dim=0)

    chamfer_distances_gt = torch.mean(min_distances_gt, dim=-1) # KxB
    all_chamfer_distances += [chamfer_distances_gt.detach().cpu().numpy()]

    models_batched = best_models.view(-1, model_dim).detach().to(inlier_device)
    features_expanded = true_coord_grid.view(B, Y, 3).view(1, 1, B, Y, 3).expand(M, K, B, Y, 3).to(inlier_device)
    features_batched = features_expanded.contiguous().view(-1, Y, 3).detach()
    occlusions_gt_batched, closest_sides_gt_batched, distances_to_cube_sides_gt_batched, _ = \
        consistency.cuboid_occlusion_torch(models_batched, features_batched)
    occlusions_gt = occlusions_gt_batched.view(M, K, B, Y, 6)

    distances_to_sides_gt = distances_to_cube_sides_gt_batched.view(M, K, B, Y, 6)
    occlusion_distances_gt = torch.sqrt(torch.max(torch.max(occlusions_gt * distances_to_sides_gt, dim=-1)[0], dim=0)[0])

    depth_error = (depth[bi, :, :].cpu().detach().numpy().squeeze() - gt_depth[bi].cpu().detach().numpy().squeeze()) ** 2
    depth_mse = np.mean(depth_error)

    sample_results["image"] = image.cpu().detach().numpy()
    sample_results["depth"] = depth.cpu().detach().numpy()
    sample_results["depth_normalised"] = depth_normalised.cpu().detach().numpy()
    sample_results["depth_mse"] = depth_mse
    sample_results["all_best_models"] = all_best_models.cpu().detach().numpy()
    sample_results["all_num_primitives"] = all_num_primitives.cpu().detach().numpy()
    sample_results["best_models"] = best_models.cpu().detach().numpy()
    sample_results["best_num_primitives"] = best_num_primitives.cpu().detach().numpy()
    sample_results["best_outer_hypos"] = best_outer_hypos.cpu().detach().numpy()

    if opt.save_all:
        sample_results["probability_maps"] = all_probs.cpu().detach().numpy()
        sample_results["selection_probabilities"] = all_q_probs.cpu().detach().numpy()
        sample_results["distances_to_sides_gt"] = distances_to_sides_gt.cpu().detach().numpy()
        sample_results["occlusions_gt"] = occlusions_gt.cpu().detach().numpy()
        sample_results["chamfer_distances_gt"] = chamfer_distances_gt.cpu().detach().numpy()
        sample_results["occlusion_distances_gt"] = occlusion_distances_gt.cpu().detach().numpy()
        sample_results["distances_gt"] = distances_gt.cpu().detach().numpy()
        sample_results["min_distances_gt"] = min_distances_gt.cpu().detach().numpy()

    for ki in range(K):
        lists_of_chamfer_distances[ki] += [chamfer_distances_gt.detach().cpu().numpy()[ki, 0]]

        min_distances_gt_numpy = min_distances_gt[ki].cpu().detach().numpy().squeeze()
        mean_chamfer = np.mean(min_distances_gt_numpy)
        occlusion_distances_gt_numpy = occlusion_distances_gt[ki].cpu().detach().numpy().squeeze()
        min_distances_gt_numpy = np.maximum(min_distances_gt_numpy, occlusion_distances_gt_numpy)
        mean_distance = np.mean(min_distances_gt_numpy)
        print("mean OA distance: %.3f" % mean_distance)
        lists_of_oa_distances[ki] += [mean_distance]
        all_distances_sorted = np.sort(min_distances_gt_numpy)
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
        sample_auc_values["mean_chamfer"] = mean_chamfer

        auc_values["mean"] += [mean_distance]
        auc_values["mean_chamfer"] += [mean_chamfer]

        sample_results["sample_%d" % ki] = {"chamfer": chamfer_distances_gt[ki].squeeze().detach().cpu().numpy(),
                                            "auc": sample_auc_values, "inlier_range": inlier_range,
                                            "all_distances_sorted": all_distances_sorted}

    if eval_folder is not None:
        target_file = os.path.join(eval_folder, "sample_%04d_%04d.pkl" % (image_idx, file_indices[0].item()))
        pickle.dump(sample_results, open(target_file, "wb"))

    single_time = time.time() - single_start
    print("time elapsed: %.2f s" % single_time)

    depth_errors = gt_depth[bi].cpu().detach().numpy().squeeze() - depth[bi, :, :].cpu().detach().numpy().squeeze()
    depth_errors *= depth_errors

    all_probs_np = all_probs[best_outer_hypos].cpu().detach().numpy().squeeze()
    all_q_probs_np = all_q_probs[best_outer_hypos].cpu().detach().numpy().squeeze()

    all_weighted_mses = []

    gray_image = rgb2gray(image[bi].cpu().detach().numpy().squeeze())
    gray_im_scaled = (gray_image - 0.5) * 0.25

    for mi in range(M):

        probs = all_probs_np[mi, :]
        q_probs = all_q_probs_np[mi, :]

        weighted_mses = []

        for qi in range(Q):

            probs_ = probs[qi]
            probs_ = probs_ / np.max(probs_)
            probs_ = skimage.transform.resize(probs_, gray_im_scaled.shape)

            weighted_depth_errors = depth_errors * (probs_)
            weighted_mse = np.sum(depth_errors) / np.sum(probs_)
            q = q_probs[qi]

            weighted_mses += [weighted_mse * q]
        all_weighted_mses += [np.sum(weighted_mses)]

    mean_weighted_mse = np.mean(all_weighted_mses)
    list_of_mean_weighted_mses += [mean_weighted_mse]

    if opt.visualise:

        all_probs_np = all_probs[best_outer_hypos].cpu().detach().numpy().squeeze()
        all_q_probs_np = all_q_probs[best_outer_hypos].cpu().detach().numpy().squeeze()
        all_best_inliers_estm_np = all_best_inliers_estm_tensor[best_outer_hypos].view(M, H_, W_).cpu().detach().numpy()
        all_best_inliers_gt_np = all_best_inliers_gt_tensor[best_outer_hypos].view(M, H_, W_).cpu().detach().numpy()
        all_best_inliers_estm_torch = all_best_inliers_estm_tensor[best_outer_hypos].view(M, 1, H_, W_)
        all_best_inliers_estm_torch = torch.nn.functional.interpolate(all_best_inliers_estm_torch,
                                                                      size=(H, W)).squeeze()
        all_best_inliers_gt_torch = all_best_inliers_gt_tensor[best_outer_hypos].view(M, 1, H_, W_)
        all_best_inliers_gt_torch = torch.nn.functional.interpolate(all_best_inliers_gt_torch, size=(H, W)).squeeze()

        colours = ['#000000', '#e6194b', '#4363d8', '#aaffc3', '#911eb4', '#46f0f0', '#f58231', '#3cb44b', '#f032e6',
                   '#008080', '#bcf60c', '#fabebe', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3']

        points = depth[bi].cpu().numpy()

        fig = plt.figure()
        plt.axis('off')
        # plt.tight_layout(pad=0.2)
        subplot_id = 1

        Vsp = np.maximum(4, Q+2)
        Hsp = M+1

        ax0a = fig.add_subplot(Vsp, M+1, 1)
        ax0a.axis('off')
        ax0a.imshow(depth[bi, :, :].cpu().detach().numpy().squeeze())

        ax0b = fig.add_subplot(Vsp, M+1, M+2)
        ax0b.imshow(depth_error)
        ax0b.set_title("MSE: %.3f" % depth_mse)
        ax0b.axis('off')

        ax0c = fig.add_subplot(Vsp, Hsp, 2*(M+1)+1)
        ax0c.axis('off')
        ax0c.imshow(gt_depth[bi].cpu().detach().numpy().squeeze())

        ax0d = fig.add_subplot(Vsp, Hsp, 3*(M+1)+1)
        ax0d.axis('off')
        ax0d.imshow(image[bi].cpu().detach().numpy().squeeze())

        gray_image = rgb2gray(image[bi].cpu().detach().numpy().squeeze())
        gray_im_scaled = (gray_image - 0.5) * 0.25

        for mi in range(M):

            points = estm_coord_flat
            probs = all_probs_np[mi, :]
            q_probs = all_q_probs_np[mi, :]

            inliers_estm = all_best_inliers_estm_torch[mi]
            inliers_gt = all_best_inliers_gt_torch[mi]

            for qi in range(Q):
                ax1 = fig.add_subplot(Vsp, Hsp, qi*(M+1) + mi + 2)
                ax1.axis('off')
                probs_ = probs[qi]
                probs_ = probs_ / np.max(probs_)
                probs_ = skimage.transform.resize(probs_, gray_im_scaled.shape)

                weighted_depth_errors = depth_errors * (probs_)
                weighted_mse = np.sum(depth_errors) / np.sum(probs_)

                q = q_probs[qi]
                ax1.imshow(probs_ + gray_im_scaled)
                ax1.set_title("%.3f" % q)

            ax2 = fig.add_subplot(Vsp, Hsp, Q*(Hsp) + mi + 2)
            ax3 = fig.add_subplot(Vsp, Hsp, (Q+1)*Hsp + mi + 2)
            ax2.axis('off')
            ax3.axis('off')

            colour_list = []

            inliers_estm = inliers_estm.detach().cpu().numpy()
            inliers_gt = inliers_gt.detach().cpu().numpy()

            ax2.imshow(inliers_estm + gray_im_scaled, vmin=-1.125, vmax=1.125)
            ax2.set_title("estimated inliers")
            ax3.imshow(inliers_gt + gray_im_scaled, vmin=-1.125, vmax=1.125)
            ax3.set_title("true inliers")

        plt.show()

    image_idx += 1

print("\nMean L2-Distances:")
chamfer_per_run = []
for ki in range(K):
    chamfer_mean = np.mean(lists_of_chamfer_distances[ki])
    chamfer_stdd = np.std(lists_of_chamfer_distances[ki])
    print("- run %d : %.6f (%.6f)" % (ki, chamfer_mean, chamfer_stdd))
    chamfer_per_run += [chamfer_mean]
print("-- per run mean: %.6f" % np.mean(chamfer_per_run))
print("-- per run stdd: %.6f" % np.std(chamfer_per_run))

print("\nMean Occlusion-Aware Distances:")
chamfer_per_run = []
for ki in range(K):
    chamfer_mean = np.mean(lists_of_oa_distances[ki])
    chamfer_stdd = np.std(lists_of_oa_distances[ki])
    print("- run %d : %.6f (%.6f)" % (ki, chamfer_mean, chamfer_stdd))
    chamfer_per_run += [chamfer_mean]
print("-- per run mean: %.6f" % np.mean(chamfer_per_run))
print("-- per run stdd: %.6f" % np.std(chamfer_per_run))

list_of_all_distances = []
for ki in range(len(lists_of_all_distances)):
    list_of_all_distances += lists_of_all_distances[ki]

print("\nAUC for Occlusion-Aware Distances:")
for max_distance in max_auc_distances:
    max_value = max_distance / 100.
    print("- AUC at %d" % max_distance)
    auc_per_run_list = []
    for ki in range(K):
        num_distances = len(lists_of_all_distances[ki])
        all_distances_sorted = np.sort(np.array(lists_of_all_distances[ki]))
        inlier_range = np.arange(num_distances).astype(np.float32) * 1. / num_distances

        x = np.append(all_distances_sorted[np.where(all_distances_sorted < max_value)], max_value)
        y = inlier_range[np.where(all_distances_sorted < max_value)]
        if y.size > 2:
            y = np.append(y, y[-1])

            auc = sklearn.metrics.auc(x, y) / max_value
        else:
            auc = 0
        print("-- run %d : %.6f" % (ki, auc))
        auc_per_run_list += [auc]
    print("--- mean: %.6f" % np.mean(auc_per_run_list))
    print("--- stdd: %.6f" % np.std(auc_per_run_list))

    num_distances = len(list_of_all_distances)
    all_distances_sorted = np.sort(np.array(list_of_all_distances))
    inlier_range = np.arange(num_distances).astype(np.float32) * 1. / num_distances

    x = np.append(all_distances_sorted[np.where(all_distances_sorted < max_value)], max_value)
    y = inlier_range[np.where(all_distances_sorted < max_value)]
    if y.size > 2:
        y = np.append(y, y[-1])
        auc = sklearn.metrics.auc(x, y) / max_value
    else:
        auc = 0

    print("---- overall AUC: %.6f" % auc)


total_time = time.time() - total_start
print("\ntime elapsed: %.1f s (%.2f s)" % (total_time, total_time*1./len(valset_loader)))

all_results = {}

all_results["auc_values"] = auc_values
all_results["best_auc_values"] = best_auc_values
all_results["best_auc_value_idx"] = best_auc_value_idx
all_results["all_chamfer_distances"] = all_chamfer_distances
all_results["lists_of_all_distances"] = lists_of_all_distances
all_results["lists_of_chamfer_distances"] = lists_of_chamfer_distances

if eval_folder is not None:
    target_file = os.path.join(eval_folder, "results.pkl")
    pickle.dump(all_results, open(target_file, "wb"))