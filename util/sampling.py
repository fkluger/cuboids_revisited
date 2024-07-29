import torch
import numpy as np
import scipy.spatial
import time


def get_choices_in_hull(choices, size=(60, 80)):
    H, W = size

    coords_2d = np.stack(np.meshgrid(range(W), range(H)), axis=-1).reshape((-1, 2))
    choices_unraveled = torch.stack([choices % W, choices // W], dim=-1)

    P, S, K, B, Y = choices.size()

    hull_choices = -torch.ones((P, S, K, B, W*H), device=choices.device, dtype=torch.int64)
    # hull_choices = torch.ones((P, S, K, B, W*H), device=choices.device, dtype=torch.int64)

    for pi in range(P):
        for si in range(S):
            for ki in range(K):
                for bi in range(B):
                    hull_points = choices_unraveled[pi, si, ki, bi]
                    hull_points = hull_points.detach().cpu().numpy()
                    try:
                        # raise Exception
                        triang = scipy.spatial.Delaunay(hull_points)
                        indices_in_hull = np.asarray(triang.find_simplex(coords_2d) >= 0).nonzero()
                        points_in_hull_2d = coords_2d[indices_in_hull]
                        N = points_in_hull_2d.shape[0]
                        points_in_hull_2d = points_in_hull_2d[:, 1]*W + points_in_hull_2d[:, 0]
                        hull_choices[pi, si, ki, bi, :N] = torch.from_numpy(points_in_hull_2d).to(choices.device)
                        del triang
                    except:
                        hull_choices[pi, si, ki, bi, :Y] = choices[pi, si, ki, bi]


    return hull_choices


def sample_model_pool_multiple_parallel_batched(data, cardinality, primitive_optimiser, probs=None, sel_probs=None,
                                                min_prob=0., choice_vecs=None, sel_choice_vecs=None,
                                                **kwargs):

    P = probs.size(0)
    S = probs.size(1)
    K = probs.size(2)
    B = probs.size(3)
    Q = probs.size(4)
    R = probs.size(5)
    Y = probs.size(6)

    if choice_vecs is None:
        choice_vecs = torch.zeros((P, S, K, B, Q, R, Y), device=data.device, dtype=torch.uint8)
    if sel_choice_vecs is None:
        sel_choice_vecs = torch.zeros((P, S, K, B, Q), device=data.device, dtype=torch.uint8)

    if R == 1:
        probs_batched = probs.contiguous().view(P*S*K*B*Q, Y)
        choice_weights = probs_batched + min_prob
        choice_batched = torch.multinomial(choice_weights, cardinality, replacement=True)
        choices = choice_batched.view(P, S, K, B, Q, 1, cardinality).detach()

        if Q == 1:
            choice_vecs.scatter_(6, choices, torch.ones(choices.size(), dtype=torch.uint8, device=data.device))
            choices = choices.squeeze(4).squeeze(4)
        else:
            choices = choices.squeeze(5)
            sel_probs_batched = sel_probs.contiguous().view(P * S * K * B, Q)
            sel_choice_weights = sel_probs_batched + min_prob
            sel_choice_batched = torch.multinomial(sel_choice_weights, 1, replacement=True)
            sel_choices = sel_choice_batched.view(P, S, K, B, 1).detach()
            sel_choice_vecs.scatter_(4, sel_choices,
                                     torch.ones(sel_choices.size(), dtype=torch.uint8, device=data.device))

            sel_choice_vecs_ = sel_choice_vecs.unsqueeze(-1).unsqueeze(-1)
            choice_vecs = choice_vecs * sel_choice_vecs_

            sel_choices = sel_choices.view(P, S, K, B, 1, 1).expand(P, S, K, B, 1, cardinality)
            choices = torch.gather(choices, 4, sel_choices)
            choices = choices.squeeze(4)

        mask = None
        models, residual, residuals = primitive_optimiser.fit(data, choices, mask, **kwargs)

    else:
        probs_batched = probs.contiguous().view(P * S * K * B * Q * R, Y)
        choice_weights = probs_batched + min_prob
        choice_batched = torch.multinomial(choice_weights, 1, replacement=True)
        choices = choice_batched.view(P, S, K, B, Q, R, 1).detach()

        if Q == 1:
            choice_vecs.scatter_(6, choices, torch.ones(choices.size(), dtype=torch.uint8, device=data.device))
            choices = choices.squeeze(4).squeeze(-1)
        else:
            choices = choices.squeeze(6)
            sel_probs_batched = sel_probs.contiguous().view(P * S * K * B, Q)
            sel_choice_weights = sel_probs_batched + min_prob
            sel_choice_batched = torch.multinomial(sel_choice_weights, 1, replacement=True)
            sel_choices = sel_choice_batched.view(P, S, K, B, 1).detach()
            sel_choice_vecs.scatter_(4, sel_choices,
                                     torch.ones(sel_choices.size(), dtype=torch.uint8, device=data.device))

            sel_choice_vecs_ = sel_choice_vecs.unsqueeze(-1).unsqueeze(-1)
            choice_vecs = choice_vecs * sel_choice_vecs_

            sel_choices = sel_choices.view(P, S, K, B, 1, 1).expand(P, S, K, B, 1, cardinality)
            choices = torch.gather(choices, 4, sel_choices)
            choices = choices.squeeze(4)

        models, residual, residuals = primitive_optimiser.fit(data, choices, **kwargs)

    del probs_batched, choice_weights, choice_batched

    return models, choice_vecs, choices, sel_choice_vecs, residual, residuals


def count_inliers_batched(data, models, inlier_fun, outlier_fun, occlusion_fun, prev_distances=None,
                          num_data=None, prev_occluded_distances=None, occlusion_penalty=True, occlusion_distance=False,
                          **kwargs):

    # models: PxSxKxBx9
    # gt_data: BxY

    P = models.size(0)
    S = models.size(1)
    K = models.size(2)
    B = models.size(3)
    Y_ = data.size(1)

    models_batched = models.view(P*S*K*B, -1)
    models_ = models_batched.view(P, S, K, B, -1)

    # distances = consistency_fun(models_, data, **kwargs)
    # inliers = inlier_fun(distances)

    features_expanded = data.view(1, 1, 1, B, Y_, 3).expand(P, S, K, B, Y_, 3).to(models.device)
    features_batched = features_expanded.contiguous().view(-1, Y_, 3)
    occlusions_batched, distances_to_cube_sides_batched = occlusion_fun(models_batched, features_batched)
    occlusions = occlusions_batched.view(P, S, K, B, Y_, 6)
    distances_to_sides = distances_to_cube_sides_batched.view(P, S, K, B, Y_, 6)
    distances = distances_to_sides.min(-1)[0]
    occluded_distances = torch.max(occlusions * distances_to_sides, dim=-1)[0]
    # occluded_distances = torch.max(distances, occluded_distances)
    if not (prev_occluded_distances is None):
        occluded_distances = \
            torch.max(occluded_distances, prev_occluded_distances)
    if not (prev_distances is None):
        distances = \
            torch.min(distances, prev_distances)

    f_io = inlier_fun(distances_to_sides) - occlusions * outlier_fun(distances_to_sides)
    min_f_io = torch.min(f_io, dim=-1)[0]
    max_f_io = torch.max(f_io, dim=-1)[0]

    inliers_oa = torch.where(min_f_io < 0, min_f_io, max_f_io)

    if not occlusion_penalty:
        inliers_oa = torch.clamp(inliers_oa, min=0)

    occlusion_aware_distances = torch.max(occluded_distances, distances)
    if occlusion_distance:
        inliers = inlier_fun(occlusion_aware_distances)
    else:
        inliers = inlier_fun(distances)

    if num_data is not None:
        for bi in range(B):
            inliers[:, :, :, bi, num_data[bi]:] = 0

    return inliers, distances, inliers_oa, occluded_distances, occlusion_aware_distances
