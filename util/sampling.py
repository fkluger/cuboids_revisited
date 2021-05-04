import torch
import torchgeometry as tgm


def sample_model_pool_multiple_parallel_batched(data, cardinality, primitive_optimiser, probs=None, sel_probs=None,
                                                min_prob=0., choice_vecs=None, sel_choice_vecs=None, **kwargs):

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

        models, residual = primitive_optimiser.fit(data, choices, **kwargs)

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

        models, residual = primitive_optimiser.fit(data, choices, **kwargs)

    del probs_batched, choice_weights, choice_batched

    return models, choice_vecs, choices, sel_choice_vecs, residual


def count_inliers_batched(data, models, inlier_fun, consistency_fun, occlusion_fun,
                          num_data=None, prev_occluded_distances=None, **kwargs):

    # models: PxSxKxBx9
    # gt_data: BxY

    P = models.size(0)
    S = models.size(1)
    K = models.size(2)
    B = models.size(3)
    Y_ = data.size(1)

    models_batched = models.view(P*S*K*B, -1)
    models_ = models_batched.view(P, S, K, B, -1)

    distances = consistency_fun(models_, data, **kwargs)
    inliers = inlier_fun(distances)

    features_expanded = data.view(1, 1, 1, B, Y_, 3).expand(P, S, K, B, Y_, 3).to(models.device)
    features_batched = features_expanded.contiguous().view(-1, Y_, 3)
    occlusions_batched, closest_sides_batched, distances_to_cube_sides_batched, _ = \
        occlusion_fun(models_batched, features_batched)
    occlusions = occlusions_batched.view(P, S, K, B, Y_, 6)
    distances_to_sides = distances_to_cube_sides_batched.view(P, S, K, B, Y_, 6)
    if prev_occluded_distances is None:
        occluded_distances = torch.max(occlusions * distances_to_sides, dim=-1)[0]
    else:
        occluded_distances = \
            torch.max(torch.max(occlusions * distances_to_sides, dim=-1)[0], prev_occluded_distances)

    f_io = inlier_fun(distances_to_sides) - occlusions * (1-inlier_fun(distances_to_sides))
    min_f_io = torch.min(f_io, dim=-1)[0]
    max_f_io = torch.max(f_io, dim=-1)[0]
    inliers_oa = torch.where(min_f_io < 0, min_f_io, max_f_io)

    if num_data is not None:
        for bi in range(B):
            inliers[:, :, :, bi, num_data[bi]:] = 0

    return inliers, distances, inliers_oa, occluded_distances
