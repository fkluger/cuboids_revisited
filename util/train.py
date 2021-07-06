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
    elif depth_model["name"] == "gt":
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
    elif depth_model["name"] == "gt":
        return
    else:
        assert False, "unknown depth model: %s" % depth_model["name"]


def save_consac_model(consac, ckpt_dir, epoch, is_best=False):
    torch.save(consac["model"].state_dict(), '%s/consac_weights_%06d.net' % (ckpt_dir, epoch))
    torch.save(consac["optimizer"].state_dict(), '%s/consac_optimizer_%06d.net' % (ckpt_dir, epoch))
    if is_best:
        torch.save(consac["model"].state_dict(), '%s/consac_weights_best.net' % ckpt_dir)
        torch.save(consac["optimizer"].state_dict(), '%s/consac_optimizer_best.net' % ckpt_dir)


def compute_losses(opt, dimensions, all_inlier_counts_estm, all_inlier_counts_gt, all_oa_distances_gt, best_multi_hypos, all_losses, all_losses_per_model):

    P = dimensions["P"]
    M = dimensions["M"]
    S = dimensions["S"]
    K = dimensions["K"]
    B = dimensions["B"]
    Y_ = dimensions["Y_"]

    all_losses_per_model_list = []

    inlier_counts_estm = \
        torch.gather(all_inlier_counts_estm, 0, best_multi_hypos.view(1, 1, 1, K, B).expand(1, M, S, K, B))
    inlier_counts_estm = inlier_counts_estm.squeeze(0)  # M x S x K x B

    inlier_counts_gt = \
        torch.gather(all_inlier_counts_gt, 0, best_multi_hypos.view(1, 1, 1, K, B).expand(1, M, S, K, B))
    inlier_counts_gt = inlier_counts_gt.squeeze(0)  # M x S x K x B

    for ki in range(0, K):

        for mi in range(opt.instances):

            if mi == 0:
                inlier_increase_estm = inlier_counts_estm[mi, :, ki]
                inlier_increase_gt = inlier_counts_gt[mi, :, ki]
            else:
                inlier_increase_estm = inlier_counts_estm[mi, :, ki] - inlier_counts_estm[mi-1, :, ki].max(0, True)[0]
                inlier_increase_gt = inlier_counts_gt[mi, :, ki] - inlier_counts_gt[mi-1, :, ki].max(0, True)[0]

            selection_weights = torch.nn.functional.softmax(inlier_increase_estm / Y_ * 10, dim=0)

            inlier_losses = torch.sum(inlier_increase_gt * selection_weights, dim=0)

            expected_losses = -inlier_losses

            all_losses_per_model[mi, ki, :] = (expected_losses * 1. / Y_)
            all_losses_per_model_list += [(expected_losses.mean() * 1. / Y_)]

        all_losses[ki, :] = all_losses_per_model.mean(0)[ki].detach()

    return all_losses_per_model_list


def backward_pass(opt, consac_model, consac_optimizer, feature_optimizer, all_losses_per_model_list, data, states,
                  neg_inliers, all_grads, all_sel_grads, depth_mse, dimensions, devices, primitive_optimiser):

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

    P, M, K, B, H, W, _ = states.size()
    B, _, _, C = data.size()
    data_expanded = data.view(1, 1, 1, B, H, W, C).expand(P, M, K, B, H, W, C)
    data_and_state = torch.cat([data_expanded, states], dim=-1)

    Q_ = Q

    fitting_device = devices[0]
    consac_device = devices[1]
    depth_device = devices[2]

    grads_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    # consac_optimizer.zero_grad()

    primitive_optimiser.zero_grad()
    if opt.train_depth and feature_optimizer is not None:
        feature_optimizer.zero_grad()

    if (opt.train_depth and feature_optimizer is not None):
        loss = torch.stack(all_losses_per_model_list).mean()
        torch.autograd.backward(loss)

    if opt.train_depth and feature_optimizer is not None:
        feature_optimizer.step()

    if opt.train_mse and feature_optimizer is not None:
        feature_optimizer.zero_grad()

        torch.autograd.backward(depth_mse, retain_graph=True)
        feature_optimizer.step()

    if opt.train_consac:
        consac_optimizer.zero_grad()

        for bi in range(B):

            output_list = []
            grad_list = []

            segments_and_selection_batched = \
                data_and_state[:, :, :, bi].view((-1, H, W, data_dim)).transpose(1, 3).transpose(3, 2).detach()

            segments_and_selection_batched_ = segments_and_selection_batched.to(consac_device)
            log_probs_batched, log_q_batched = consac_model(segments_and_selection_batched_)
            del segments_and_selection_batched_

            grads_batched = all_grads[:, :, :, bi].view((-1, Q * R, H_, W_)).to(consac_device).detach()
            sel_grads_batched = all_sel_grads[:, :, :, bi].view((-1, Q)).to(consac_device).detach()

            if opt.loss_clamp > 0:
                grads_batched = torch.clamp(grads_batched, max=opt.loss_clamp, min=-opt.loss_clamp)

            if opt.train_consac:
                output_list += [log_probs_batched[:, :Q].contiguous()]
                grad_list += [grads_batched.contiguous()]

            if log_q_batched is not None and opt.train_consac:
                output_list += [log_q_batched.contiguous()]
                grad_list += [sel_grads_batched.contiguous()]

                if opt.minimise_corr > 0:
                    log_q = log_q_batched.view(P, M, K, Q)
                    log_probs = log_probs_batched.view(
                        P, M, K, Q_, R, Y_)
                    probs = torch.softmax(log_probs, dim=-1)
                    q_probs = torch.softmax(log_q, dim=-1)
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
                        entropy = torch.distributions.categorical.Categorical(probs=q_probs.view(P * M * K, Q)).entropy()
                        output_list += [-entropy.mean()]
                        grad_list += [torch.tensor(opt.maximise_second_entropy).to(entropy.device)]

            log_probs = log_probs_batched[:, :Q].view(
                P, M, K, Q, R, Y_)
            probs = torch.softmax(log_probs, dim=-1)
            if opt.max_prob_loss > 0 and opt.train_consac:

                max_probs, _ = torch.max(probs, dim=-1, keepdim=True)
                del _
                probs = probs / torch.clamp(max_probs, min=1e-8)
                max_prob_loss = torch.clamp(probs - neg_inliers[:, 0:M, :, bi].unsqueeze(3).unsqueeze(4).to(consac_device).detach(),
                                            min=0).contiguous().to(consac_device)
                max_prob_grad = opt.max_prob_loss * torch.ones_like(max_prob_loss, device=consac_device).contiguous()

                output_list += [max_prob_loss]
                grad_list += [max_prob_grad]

            if len(output_list) > 0:
                torch.autograd.backward(output_list, grad_list)

            for var in output_list:
                del var
            for var in grad_list:
                del var

            del segments_and_selection_batched

        consac_optimizer.step()

    primitive_optimiser.backward_step()

    torch.set_grad_enabled(grads_enabled)


def gather_oa_distances(best_outer_hypos, all_best_oa_distances_gt):

    P, K, B, Y_ = all_best_oa_distances_gt.size()
    max_distances = torch.gather(all_best_oa_distances_gt, 0,
                                 best_outer_hypos.view(1, K, B, 1).expand(1, K, B, Y_).to(all_best_oa_distances_gt.device))
    return max_distances.cpu().detach().numpy().flatten()


def calc_auc_values(distances, max_auc_distances=(5., 10., 20., 50.)):

    if type(distances) is list:
        distances = np.concatenate(distances)

    all_distances_sorted = np.sort(distances)
    inlier_range = np.arange(distances.size).astype(np.float32) * 1. / distances.size

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

    return sample_auc_values
