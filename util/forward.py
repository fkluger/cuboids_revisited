from util.misc import *
from util import sampling, consistency
import torch
import pickle
import numpy as np


def estimate_depth(opt, images, image_mean, depth_model, dimensions, devices, depth_mean, depth_stdd, gt_depth=None,
                   write_cache=False, read_cache=False):

    H = dimensions["H"]
    W = dimensions["W"]
    device = devices[2]

    mean = torch.tensor([0.485, 0.456, 0.406], device=None).view(1, 1, 1, 3).to(device)
    stdd = torch.tensor([0.229, 0.224, 0.225], device=None).view(1, 1, 1, 3).to(device)
    if not depth_model["name"] == "gt":
        images = (images.to(device) - mean) / stdd
        images = images.transpose(1, 3).transpose(2, 3)
        images_scaled = torch.nn.functional.interpolate(images.to(device),
                                                        size=(depth_model["height"], depth_model["width"]))
        images_scaled = images_scaled.transpose(2, 3).transpose(1, 3)
    else:
        images_scaled = None

    if write_cache or read_cache:
        cache_file = depth_cache_path(opt, images.cpu().detach().numpy())

    if read_cache and os.path.exists(cache_file):
        # cache_file = depth_cache_path(opt, images.cpu().detach().numpy(), create=False)
        depth, depth_normalised, depth_mse = pickle.load(open(cache_file, 'rb'))
        depth = torch.from_numpy(depth.astype(np.float32))
        depth_normalised = torch.from_numpy(depth_normalised.astype(np.float32))
        depth_mse = torch.from_numpy(depth_mse.astype(np.float32))
    else:

        if depth_model["name"] == "bts":

            focal = torch.tensor([1.]).to(device)
            images_ = images_scaled.transpose(1, 3).transpose(2, 3)

            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est, depth_est_normed = depth_model["model"](images_, focal)
            del lpg2x2, lpg4x4, lpg8x8, reduc1x1

            # depth_normalised = depth_est_normed.transpose(2, 3).transpose(1, 3).to(device)
            depth_normalised = torch.transpose(torch.transpose(depth_est_normed, 2, 3), 1, 3).to(device)

            # depth_padded = depth_est

            # depth = depth_padded.transpose(2, 3).transpose(1, 3).to(device)
            depth = torch.transpose(torch.transpose(depth_est, 2, 3), 1, 3).to(device)

        elif depth_model["name"] == "dummy":
            depth = depth_model["model"] * images_scaled[:,:,:,0].unsqueeze(-1)
            depth_normalised = depth_model["model"] * images_scaled[:,:,:,0].unsqueeze(-1)

        elif depth_model["name"] == "gt":
            assert (gt_depth is not None), "No ground truth depth given!"

            depth = gt_depth.unsqueeze(-1)

            depth_normalised = depth - depth_mean
            depth_normalised = depth_normalised / depth_stdd

        else:
            assert False, "unknown depth model: %s" % depth_model["name"]

        squared_depth_error = (depth.squeeze(-1) - gt_depth.to(depth.device)) ** 2
        depth_mse = torch.mean(squared_depth_error)

    if write_cache:
        pickle.dump((depth.cpu().numpy(), depth_normalised.cpu().numpy(), depth_mse.cpu().numpy()), open(cache_file, "wb"))

    return depth, depth_normalised, depth_mse, images_scaled


def estimate_sampling_weights(opt, dims, devices, data, state, consac, previous_inliers=None):

    fitting_device = devices[0]
    consac_device = devices[1]
    depth_device = devices[2]

    P, K, B, H, W, _ = state.size()
    B, _, _, C = data.size()
    data_expanded = data.view(1, 1, B, H, W, C).expand(P, K, B, H, W, C)
    data_and_state = torch.cat([data_expanded, state], dim=-1)

    if opt.seqransac:
        # assert False, "TODO: check if Seq.RANSAC still works correctly"
        if previous_inliers is None:
            previous_inliers = torch.zeros((dims["P"], dims["K"], dims["B"], dims["Y_"]), device=fitting_device)
        probs = torch.ones((dims["P"], dims["K"], dims["B"], dims["Q"], dims["R"], dims["Y_"]), device=fitting_device) \
            - torch.clamp(previous_inliers.view(dims["P"], dims["K"], dims["B"], 1, 1, dims["Y_"]), min=0)
        q = torch.ones((dims["P"], dims["K"], dims["B"], dims["Q"]), device=fitting_device)

        log_probs = torch.log(probs)
        log_q = torch.log(q)

        cur_probs = probs.unsqueeze(1). \
            expand(dims["P"], dims["S"], dims["K"], dims["B"], dims["Q"], dims["R"], dims["Y_"])
        cur_q = q.unsqueeze(1).expand(dims["P"], dims["S"], dims["K"], dims["B"], dims["Q"])

        entropy = torch.distributions.categorical.Categorical(probs=probs).entropy()
        return cur_probs, cur_q, log_probs, log_q, entropy

    # log_probs = torch.zeros((dims["P"], dims["K"], dims["B"], dims["Q"], dims["R"], dims["Y_"]), device=fitting_device)
    # log_q = torch.zeros((dims["P"], dims["K"], dims["B"], dims["Q"]), device=fitting_device)
    # for bi in range(dims["B"]):
    #
    #     segments_and_selection_batched = \
    #         data_and_state[:, :, bi].contiguous().view((-1, dims["H"], dims["W"], dims["data"])).transpose(1, 3).transpose(3, 2)
    #
    #     # with torch.no_grad():
    #     segments_and_selection_batched_ = segments_and_selection_batched.to(consac_device)
    #     log_probs_batched, log_q_batched = consac["model"](segments_and_selection_batched_)
    #     del segments_and_selection_batched_
    #
    #     B_ = log_probs_batched.size(0)
    #     log_probs[:, :, bi] = log_probs_batched[:, :dims["Q"]].view(B_, dims["Q"] * dims["R"], dims["Y_"]).\
    #         view(dims["P"], dims["K"], dims["Q"] * dims["R"], dims["Y_"]).\
    #         view(dims["P"], dims["K"], dims["Q"], dims["R"], dims["Y_"]).to(fitting_device)
    #     del log_probs_batched
    #     del segments_and_selection_batched
    #
    #     if log_q_batched is not None:
    #         log_q[:, :, bi] = log_q_batched.view(B_, dims["Q"]).view(dims["P"], dims["K"], dims["Q"]).to(fitting_device)

    segments_and_selection_batched = \
        data_and_state.contiguous().view((-1, dims["H"], dims["W"], dims["data"])).transpose(1, 3).transpose(
            3, 2)

    # with torch.no_grad():
    segments_and_selection_batched_ = segments_and_selection_batched.to(consac_device)
    log_probs_batched, log_q_batched = consac["model"](segments_and_selection_batched_)
    del segments_and_selection_batched_

    B_ = log_probs_batched.size(0)
    log_probs = log_probs_batched[:, :dims["Q"]].view(B_, dims["Q"] * dims["R"], dims["Y_"]). \
        view(dims["P"], dims["K"], dims["B"], dims["Q"] * dims["R"], dims["Y_"]). \
        view(dims["P"], dims["K"], dims["B"], dims["Q"], dims["R"], dims["Y_"]).to(fitting_device)
    del log_probs_batched
    del segments_and_selection_batched

    log_q = log_q_batched.view(B_, dims["Q"]).view(dims["P"], dims["K"], dims["B"], dims["Q"]).to(fitting_device)

    # cur_probs_ = torch.clone(log_probs)
    # for bi in range(dims["B"]):
    #     cur_probs_[:, :, bi, :] = torch.softmax(log_probs[:, :, bi, :], dim=-1)
    cur_probs_ = torch.softmax(log_probs, dim=-1)

    cur_probs = cur_probs_.unsqueeze(1).\
        expand(dims["P"], dims["S"], dims["K"], dims["B"], dims["Q"], dims["R"], dims["Y_"])

    # cur_q_ = torch.clone(log_q)
    # for bi in range(dims["B"]):
    #     cur_q_[:, :, bi, :] = torch.softmax(log_q[:, :, bi, :], dim=-1)
    cur_q_ = torch.softmax(log_q, dim=-1)
    cur_q = cur_q_.unsqueeze(1).expand(dims["P"], dims["S"], dims["K"], dims["B"], dims["Q"])

    entropy = torch.distributions.categorical.Categorical(logits=log_probs, validate_args=False).entropy()

    del cur_probs_, cur_q_

    return cur_probs, cur_q, log_probs, log_q, entropy


def generate_coordinate_grids(estimated_depth, true_coord_grid, dimensions, devices, mask, gt=False):

    H_ = dimensions["H_"]
    W_ = dimensions["W_"]
    Y_ = dimensions["Y_"]
    B = dimensions["B"]

    fitting_device = devices[0]
    consac_device = devices[1]
    depth_device = devices[2]

    if gt:
        estm_coord_grid = true_coord_grid
    else:

        estm_coord_grid = true_coord_grid.clone()

        depth_ = estimated_depth.view(estm_coord_grid.size(0), estm_coord_grid.size(1), estm_coord_grid.size(2), 1).expand(
            estm_coord_grid.size()).to(estm_coord_grid.device)
        estm_coord_grid[:, :, :, 0] = estm_coord_grid[:, :, :, 0] / estm_coord_grid[:, :, :, 2]
        estm_coord_grid[:, :, :, 1] = estm_coord_grid[:, :, :, 1] / estm_coord_grid[:, :, :, 2]
        estm_coord_grid[:, :, :, 2] = 1
        estm_coord_grid = estm_coord_grid * depth_
        del depth_

        estm_coord_grid_small = \
            torch.nn.functional.interpolate(
                estm_coord_grid.transpose(1, 2).transpose(1, 3), size=(H_, W_)).transpose(1, 2).transpose(2, 3)

        estm_coord_flat = estm_coord_grid_small.view(B, Y_, 3).to(fitting_device)

    true_coord_grid = true_coord_grid.to(fitting_device)

    true_coord_grid_small = \
        torch.nn.functional.interpolate(
            true_coord_grid.transpose(1, 2).transpose(1, 3), size=(H_, W_)).transpose(1, 2).transpose(2, 3)

    true_coord_flat = true_coord_grid_small.view(B, Y_, 3).to(fitting_device)

    if gt:
        estm_coord_flat = true_coord_flat
        estm_coord_grid_small = true_coord_grid_small

    mask_small = \
        torch.nn.functional.interpolate(mask[:, None, ...], size=(H_, W_))

    mask_flat = mask_small.view(B, Y_).to(fitting_device)

    return true_coord_grid, true_coord_grid_small, true_coord_flat, estm_coord_grid, estm_coord_grid_small, estm_coord_flat, mask_flat, mask_small


def estimate_models(opt, dimensions, estm_coord_flat, sampling_weight_maps, selection_weights, minimal_solver):

    fitting_device = minimal_solver.device

    minimal_set_size = dimensions["mss"]

    models, choices, choice_indices, sel_choices, residual, residuals = \
        sampling.sample_model_pool_multiple_parallel_batched(
            estm_coord_flat.detach().to(fitting_device), minimal_set_size,
            minimal_solver,
            probs=sampling_weight_maps.to(fitting_device),
            min_prob=opt.min_prob, choice_vecs=None,
            sel_probs=selection_weights.to(fitting_device))

    return models, choices, sel_choices, residual, residuals


def count_inliers(opt, models, features, inlier_fun, outlier_fun, prev_distances, prev_occluded_distances, prev_inliers, mode):

    fitting_device = models.device

    if opt is None:
        a_max = 3
        a_min = 0.01
    else:
        a_max = opt.a_max
        a_min = opt.a_min

    if mode == "occlusion_aware":
        occlusion_penalty = True
        occlusion_distance = False
        occlusion_aware = True
    elif mode == "occlusion_distance":
        occlusion_penalty = False
        occlusion_distance = True
        occlusion_aware = False
    elif mode == "occlusion_nopenalty":
        occlusion_penalty = False
        occlusion_distance = False
        occlusion_aware = True
    elif mode == "normal":
        occlusion_penalty = False
        occlusion_distance = False
        occlusion_aware = False
    else:
        assert False

    # consistency_fun = consistency.cuboid_distance_multidim
    occlusion_fun = consistency.cuboid_occlusion_batched

    inliers, distances, inliers_oa, occluded_distances, occlusion_aware_distances = \
        sampling.count_inliers_batched(features.to(fitting_device), models, inlier_fun, outlier_fun,
                                       occlusion_fun, a_max=a_max, a_min=a_min,
                                       occlusion_penalty=occlusion_penalty, occlusion_distance=occlusion_distance,
                                       prev_occluded_distances=prev_occluded_distances, prev_distances=prev_distances)

    if occlusion_aware:
        inliers = inliers_oa

    # if prev_distances is not None:
    #     distances = torch.min(distances, prev_distances)

    # if prev_occluded_distances is not None:
    #     occluded_distances = torch.max(occluded_distances, prev_occluded_distances)

    if prev_inliers is not None:
        min_inliers = torch.min(inliers, prev_inliers)
        max_inliers = torch.max(inliers, prev_inliers)
        inliers = torch.where(min_inliers < 0, min_inliers, max_inliers)

    return inliers, distances, occluded_distances, occlusion_aware_distances

#
# def estimate_models_and_count_inliers(opt, dimensions, devices, estm_coord_flat, true_coord_flat, cur_probs, cur_q,
#                                       inlier_fun, primitve_optimiser):
#
#     fitting_device = devices[0]
#
#     P = dimensions["P"]
#     S = dimensions["S"]
#     K = dimensions["K"]
#     B = dimensions["B"]
#     Y_ = dimensions["Y_"]
#     minimal_set_size = dimensions["mss"]
#     model_dim = dimensions["model"]
#
#     consistency_fun = consistency.cuboid_distance_multidim
#     occlusion_fun = consistency.cuboid_occlusion_batched
#
#     num_data = torch.ones(B, dtype=torch.int, device=fitting_device) * Y_
#
#     models, choices, choice_indices, sel_choices, residual = \
#         sampling.sample_model_pool_multiple_parallel_batched(
#             estm_coord_flat.detach().to(fitting_device), minimal_set_size,
#             primitve_optimiser,
#             probs=cur_probs.to(fitting_device),
#             min_prob=opt.min_prob, choice_vecs=None,
#             sel_probs=cur_q.to(fitting_device))
#
#     models = models.detach()
#
#     inliers_estm, distances_estm, inliers_oa_estm, occluded_distances_estm = \
#         sampling.count_inliers_batched(estm_coord_flat.to(fitting_device), models.detach(), inlier_fun, consistency_fun,
#                                        occlusion_fun, num_data,
#                                        a_max=opt.a_max, a_min=opt.a_min)
#
#     if not opt.no_oai_sampling:
#         inliers_estm = inliers_oa_estm
#
#     inliers_gt, distances_gt, inliers_oa_gt, occluded_distances_gt = \
#         sampling.count_inliers_batched(true_coord_flat.to(fitting_device), models, inlier_fun, consistency_fun,
#                                        occlusion_fun, num_data, a_max=opt.a_max, a_min=opt.a_min)
#
#     if not opt.no_oai_loss:
#         inliers_gt = inliers_oa_gt
#
#     del choice_indices
#
#     return inliers_estm, inliers_gt, distances_estm, distances_gt, occluded_distances_estm, occluded_distances_gt, \
#            sel_choices, choices, models, residual


def select_single_hypotheses(dims, inliers, models=None):

    cumulative_inlier_counts = torch.sum(inliers, dim=-1)
    best_hypo = torch.argmax(cumulative_inlier_counts, dim=1)

    if models is not None:

        indices = best_hypo.view(dims["P"], 1, dims["K"], dims["B"], 1).\
            expand(dims["P"], 1, dims["K"], dims["B"], dims["model"]).to(models.device)

        selected_models = \
            torch.gather(models, 1, indices).squeeze(1)

        return best_hypo, selected_models
    else:
        return best_hypo
