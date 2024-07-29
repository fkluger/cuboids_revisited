import torch
import numpy as np
import sklearn.metrics


def set_eval_depth(depth_model):
    if depth_model["name"] == "monodepth2":
        depth_model["encoder"].eval()
        depth_model["decoder"].eval()
    elif depth_model["name"] == "bts" or depth_model["name"] == "dummy_depth":
        depth_model["model"].eval()
    elif depth_model["name"] == "gt":
        return
    else:
        assert False, "unknown depth model: %s" % depth_model["name"]


def set_train_depth(depth_model):
    if depth_model["name"] == "monodepth2":
        depth_model["encoder"].train()
        depth_model["decoder"].train()
    elif depth_model["name"] == "bts" or depth_model["name"] == "dummy_depth":
        depth_model["model"].train()
    elif depth_model["name"] == "gt" or depth_model["name"] == "dummy":
        return
    else:
        assert False, "unknown depth model: %s" % depth_model["name"]


def save_depth_model(depth_model, ckpt_dir, epoch, is_best=False):
    if depth_model["name"] == "monodepth2":
        torch.save(depth_model["encoder"].state_dict(), '%s/encoder_weights_%06d.net' % (ckpt_dir, epoch))
        torch.save(depth_model["decoder"].state_dict(), '%s/decoder_weights_%06d.net' % (ckpt_dir, epoch))
        torch.save(depth_model["optimizer"].state_dict(), '%s/feature_optimizer_%06d.net' % (ckpt_dir, epoch))
        if is_best:
            torch.save(depth_model["encoder"].state_dict(), '%s/encoder_weights_best.net' % ckpt_dir)
            torch.save(depth_model["decoder"].state_dict(), '%s/decoder_weights_best.net' % ckpt_dir)
            torch.save(depth_model["optimizer"].state_dict(), '%s/feature_optimizer_best.net' % ckpt_dir)
    elif depth_model["name"] == "bts" or depth_model["name"] == "dummy_depth":
        torch.save(depth_model["model"].state_dict(), '%s/depth_weights_%06d.net' % (ckpt_dir, epoch))
        torch.save(depth_model["optimizer"].state_dict(), '%s/feature_optimizer_%06d.net' % (ckpt_dir, epoch))
        if is_best:
            torch.save(depth_model["model"].state_dict(), '%s/depth_weights_best.net' % ckpt_dir)
            torch.save(depth_model["optimizer"].state_dict(), '%s/feature_optimizer_best.net' % ckpt_dir)
    elif depth_model["name"] == "gt" or depth_model["name"] == "dummy":
        return
    else:
        assert False, "unknown depth model: %s" % depth_model["name"]


def save_consac_model(consac, ckpt_dir, epoch, is_best=False):
    torch.save(consac["model"].state_dict(), '%s/consac_weights_%06d.net' % (ckpt_dir, epoch))
    torch.save(consac["optimizer"].state_dict(), '%s/consac_optimizer_%06d.net' % (ckpt_dir, epoch))
    if is_best:
        torch.save(consac["model"].state_dict(), '%s/consac_weights_best.net' % ckpt_dir)
        torch.save(consac["optimizer"].state_dict(), '%s/consac_optimizer_best.net' % ckpt_dir)


def compute_rewards(inlier_counts, discount=1.):
    max_counts = inlier_counts.max(1)[0]
    zero_pad = torch.zeros(1, max_counts.size(1), max_counts.size(2), device=max_counts.device)
    diff_tensor = torch.cat([zero_pad, max_counts[:-1]], dim=0).unsqueeze(1)

    single_rewards = inlier_counts-diff_tensor

    accu_rewards_list = []

    for mi in range(inlier_counts.size(0)-1, -1, -1):
        rewards = single_rewards[mi]
        if len(accu_rewards_list) > 0:
            rewards = rewards + discount*accu_rewards_list[0]
        accu_rewards_list = [rewards] + accu_rewards_list

    accu_rewards = torch.stack(accu_rewards_list, dim=0)

    return accu_rewards


def compute_losses(opt, dimensions, all_inlier_counts_estm, all_inlier_counts_gt, all_best_single_hypos, best_multi_hypos,
                   all_losses, all_losses_per_model):

    P = dimensions["P"]
    M = dimensions["M"]
    S = dimensions["S"]
    K = dimensions["K"]
    B = dimensions["B"]
    Y_ = dimensions["Y_"]

    inlier_counts_estm = \
        torch.gather(all_inlier_counts_estm, 0, best_multi_hypos.view(1, 1, 1, K, B).expand(1, M, S, K, B))
    inlier_counts_estm = inlier_counts_estm.squeeze(0)  # M x S x K x B

    inlier_counts_gt = \
        torch.gather(all_inlier_counts_gt, 0, best_multi_hypos.view(1, 1, 1, K, B).expand(1, M, S, K, B))
    inlier_counts_gt = inlier_counts_gt.squeeze(0)

    best_single_hypos = torch.gather(all_best_single_hypos, 0, best_multi_hypos.view(1, 1, K, B).expand(1, M, K, B))
    best_single_hypos = best_single_hypos.squeeze(0).unsqueeze(1)

    best_inlier_counts_estm = \
        torch.gather(inlier_counts_estm, 1, best_single_hypos)
    best_inlier_counts_gt = \
        torch.gather(inlier_counts_gt, 1, best_single_hypos)

    # all_rewards = compute_rewards(inlier_counts_gt, opt.discount_factor)

    mean_loss_per_model = 0

    all_expected_rewards = torch.zeros(M, K, B).to(inlier_counts_estm.device)

    for ki in range(0, K):

        for mi in range(opt.instances):

            if mi == 0:
                inlier_increase_estm = inlier_counts_estm[mi, :, ki]
                inlier_increase_gt = inlier_counts_gt[mi, :, ki]
            else:
                inlier_increase_estm = inlier_counts_estm[mi, :, ki] - best_inlier_counts_estm[mi-1, :, ki]
                inlier_increase_gt = inlier_counts_gt[mi, :, ki] - best_inlier_counts_gt[mi-1, :, ki]

            rewards = inlier_increase_gt

            selection_weights = torch.nn.functional.softmax(inlier_increase_estm / Y_ * opt.softmax_alpha, dim=0)
            expected_rewards = torch.sum(rewards * selection_weights, dim=0)
            all_expected_rewards[mi, ki] = expected_rewards.detach()
            inlier_losses = expected_rewards * 1./Y_

            expected_losses = -inlier_losses

            all_losses_per_model[mi, ki, :] = expected_losses
            # all_losses_per_model_list += [expected_losses.mean()]
            mean_loss_per_model += expected_losses.mean()

        all_losses[ki, :] = all_losses_per_model.mean(0)[ki].detach()

    mean_loss_per_model = mean_loss_per_model * 1. / (K*opt.instances)

    return mean_loss_per_model, all_expected_rewards.cpu().numpy()


def backward_pass(opt, consac_optimizer, feature_optimizer, mean_loss_per_model, neg_inliers, all_grads,
                  all_sel_grads, depth_mse, dimensions, devices, primitive_optimiser, mean_residual, all_log_probs,
                  all_log_q):

    B = dimensions["B"]
    H = dimensions["H"]
    H_ = dimensions["H_"]
    W = dimensions["W"]
    W_ = dimensions["W_"]
    P = dimensions["P"]
    M = dimensions["M"]
    K = dimensions["K"]
    Q = dimensions["Q"]
    R = dimensions["R"]
    Y_ = dimensions["Y_"]
    data_dim = dimensions["data"]

    Q_ = Q

    fitting_device = devices[0]
    consac_device = devices[1]
    depth_device = devices[2]

    grads_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(True)

    output_list = []
    grad_list = []

    primitive_optimiser.zero_grad()
    if opt.train_depth and feature_optimizer is not None:
        feature_optimizer.zero_grad()

    if (opt.train_depth and feature_optimizer is not None) or opt.train_solver:

        loss = mean_loss_per_model

        output_list += [loss]
        grad_list += [torch.ones_like(loss).to(loss.device)]

    if opt.train_consac:
        consac_optimizer.zero_grad()

        for bi in range(B):

            log_probs_batched = all_log_probs[:, :, :, bi].view(-1, Q, H_, W_)
            log_q_batched = all_log_q[:, :, :, bi].view(-1, Q)

            grads_batched = all_grads[:, :, :, bi].view((-1, Q * R, H_, W_)).to(consac_device).detach()
            sel_grads_batched = all_sel_grads[:, :, :, bi].view((-1, Q)).to(consac_device).detach()

            if opt.loss_clamp > 0:
                grads_batched = torch.clamp(grads_batched, max=opt.loss_clamp, min=-opt.loss_clamp)
                sel_grads_batched = torch.clamp(sel_grads_batched, max=opt.loss_clamp, min=-opt.loss_clamp)

            if opt.train_consac:
                output_list += [log_probs_batched[:, :Q].contiguous()]
                grad_list += [grads_batched.contiguous()]

            if log_q_batched is not None and opt.train_consac and log_q_batched.requires_grad:
                output_list += [log_q_batched.contiguous()]
                grad_list += [sel_grads_batched.contiguous()]

                if opt.minimise_corr > 0:
                    log_probs = log_probs_batched.view(
                        P, M, K, Q_, R, Y_)
                    probs = torch.softmax(log_probs, dim=-1)
                    for q1 in range(Q_):
                        vx = probs[:, :, :, q1] - torch.mean(probs[:, :, :, q1], dim=-1, keepdim=True)
                        for q2 in range(q1 + 1, Q_):
                            vy = probs[:, :, :, q2] - torch.mean(probs[:, :, :, q2], dim=-1, keepdim=True)
                            corr = torch.sum(vx * vy, dim=-1) * torch.rsqrt(torch.sum(vx ** 2, dim=-1)) \
                                   * torch.rsqrt(torch.sum(vy ** 2, dim=-1))
                            cost = torch.mean(corr)
                            output_list += [cost]
                            grad_list += [torch.tensor(opt.minimise_corr).to(cost.device)]

                if opt.maximise_second_entropy > 0:
                    log_q = log_q_batched.view(P, M, K, Q)
                    q_probs = torch.softmax(log_q, dim=-1)
                    entropy = torch.distributions.categorical.Categorical(probs=q_probs.view(P * M * K, Q)).entropy()
                    output_list += [-entropy.mean()]
                    grad_list += [torch.tensor(opt.maximise_second_entropy).to(entropy.device)]


            log_probs = log_probs_batched[:, :Q].view(
                P, M, K, Q, R, Y_)
            probs = torch.softmax(log_probs[:, 1:], dim=-1)
            if opt.max_prob_loss > 0 and opt.train_consac:

                max_probs, _ = torch.max(probs, dim=-1, keepdim=True)
                del _
                probs = probs / torch.clamp(max_probs, min=1e-8)
                max_prob_loss = torch.clamp(probs - neg_inliers[:, :, :, bi].unsqueeze(3).unsqueeze(4).to(consac_device).detach(),
                                            min=0).contiguous().to(consac_device)
                max_prob_grad = opt.max_prob_loss * torch.ones_like(max_prob_loss, device=consac_device).contiguous()

                output_list += [max_prob_loss]
                grad_list += [max_prob_grad]

        # log_probs_batched = all_log_probs.view(-1, Q, H_, W_)
        # log_q_batched = all_log_q.view(-1, Q)
        #
        # grads_batched = all_grads.view((-1, Q * R, H_, W_)).to(consac_device).detach()
        # sel_grads_batched = all_sel_grads.view((-1, Q)).to(consac_device).detach()
        #
        # if opt.loss_clamp > 0:
        #     grads_batched = torch.clamp(grads_batched, max=opt.loss_clamp, min=-opt.loss_clamp)
        #     sel_grads_batched = torch.clamp(sel_grads_batched, max=opt.loss_clamp, min=-opt.loss_clamp)
        #
        # if opt.train_consac:
        #     output_list += [log_probs_batched[:, :Q].contiguous()]
        #     grad_list += [grads_batched.contiguous()]
        #
        # if log_q_batched is not None and opt.train_consac and log_q_batched.requires_grad:
        #     output_list += [log_q_batched.contiguous()]
        #     grad_list += [sel_grads_batched.contiguous()]
        #
        #     if opt.minimise_corr > 0:
        #         log_probs = log_probs_batched.view(
        #             P, M, K, B, Q_, R, Y_)
        #         probs = torch.softmax(log_probs, dim=-1)
        #         for q1 in range(Q_):
        #             vx = probs[:, :, :, :, q1] - torch.mean(probs[:, :, :, :, q1], dim=-1, keepdim=True)
        #             for q2 in range(q1 + 1, Q_):
        #                 vy = probs[:, :, :, :, q2] - torch.mean(probs[:, :, :, :, q2], dim=-1, keepdim=True)
        #                 corr = torch.sum(vx * vy, dim=-1) * torch.rsqrt(torch.sum(vx ** 2, dim=-1)) \
        #                        * torch.rsqrt(torch.sum(vy ** 2, dim=-1))
        #                 cost = torch.mean(corr)
        #                 output_list += [cost]
        #                 grad_list += [torch.tensor(opt.minimise_corr).to(cost.device)]
        #
        #     if opt.maximise_second_entropy > 0:
        #         log_q = log_q_batched.view(P, M, K, B, Q)
        #         q_probs = torch.softmax(log_q, dim=-1)
        #         entropy = torch.distributions.categorical.Categorical(probs=q_probs.view(P * M * K * B, Q)).entropy()
        #         output_list += [-entropy.mean()]
        #         grad_list += [torch.tensor(opt.maximise_second_entropy).to(entropy.device)]
        #
        #     if opt.minimise_overlap > 0:
        #         # log_q = log_q_batched.view(P, M, K, Q)
        #         log_probs = log_probs_batched.view(
        #             P, M, K, B, Q_, R, Y_)
        #         probs = torch.softmax(log_probs, dim=-1)
        #         # q_probs = torch.softmax(log_q, dim=-1)
        #         for q1 in range(Q_):
        #             vx = probs[:, :, :, :, q1]
        #             for q2 in range(q1 + 1, Q_):
        #                 vy = probs[:, :, :, :, q2]
        #                 corr = torch.sum(vx * vy, dim=-1)* torch.rsqrt(torch.sum(vx ** 2, dim=-1)) \
        #                        * torch.rsqrt(torch.sum(vy ** 2, dim=-1))
        #                 cost = torch.mean(corr)
        #                 output_list += [cost]
        #                 grad_list += [torch.tensor(opt.minimise_overlap).to(cost.device)]
        #
        #
        # log_probs = log_probs_batched[:, :Q].view(
        #     P, M, K, B, Q, R, Y_)
        # probs = torch.softmax(log_probs[:, 1:], dim=-1)
        # if opt.max_prob_loss > 0 and opt.train_consac:
        #
        #     max_probs, _ = torch.max(probs, dim=-1, keepdim=True)
        #     del _
        #     probs = probs / torch.clamp(max_probs, min=1e-8)
        #     max_prob_loss = torch.clamp(probs - neg_inliers[:, :, :, :].unsqueeze(3).unsqueeze(4).to(consac_device).detach(),
        #                                 min=0).contiguous().to(consac_device)
        #     max_prob_grad = opt.max_prob_loss * torch.ones_like(max_prob_loss, device=consac_device).contiguous()
        #
        #     output_list += [max_prob_loss]
        #     grad_list += [max_prob_grad]

    if len(output_list) > 0:
        # torch.autograd.backward(output_list, grad_list)

        bwi = 0
        for o, g in zip(output_list, grad_list):
            torch.autograd.backward(o.to(g.device), g, retain_graph=True)
            bwi += 1

    for var in output_list:
        del var
    for var in grad_list:
        del var

    if opt.train_consac:
        consac_optimizer.step()
    primitive_optimiser.backward_step()

    if opt.train_depth and feature_optimizer is not None:
        feature_optimizer.step()

    torch.set_grad_enabled(grads_enabled)


def gather_oa_distances(best_outer_hypos, all_best_oa_distances_gt):

    P, K, B, Y_ = all_best_oa_distances_gt.size()
    max_distances = torch.gather(all_best_oa_distances_gt, 0,
                                 best_outer_hypos.view(1, K, B, 1).expand(1, K, B, Y_).to(all_best_oa_distances_gt.device))
    return max_distances.cpu().detach().numpy().flatten()


def calc_auc_values(distances, max_auc_distances=(5., 10., 20., 50., 100., 200., 500.)):

    if type(distances) is list:
        distances = np.concatenate(distances)

    all_distances_sorted = np.sort(distances)
    inlier_range = np.arange(distances.size).astype(np.float32) * 1. / distances.size

    mean_distance = np.mean(distances)

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

    return sample_auc_values, mean_distance
