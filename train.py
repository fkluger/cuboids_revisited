from util.train import *
from util.options import *
from util.forward import *
from util.fitting import *
from util.initialisation import *
from datasets.nyu_depth.nyu_depth import NYURGBDataset
from datasets.smh import SMHDataset
import networks.layers
import numpy as np
import platform
import random
import torch
import sys
import time
import wandb

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

opt = get_options()

if opt.resume is not None:
    opt = load_opts_for_resume(opt)

gettrace = getattr(sys, 'gettrace', None)
if gettrace():
    opt.debugging = True
    torch.autograd.set_detect_anomaly(True)

ckpt_dir, log, loss_log_writer, loss_log, session_id = get_log_and_checkpoint_directory(opt)

wandb.init(project="cuboid_fitting", dir=opt.wandb_dir, mode=opt.wandb_mode, group=opt.wandb_group)
wandb.config.update(vars(opt))
wandb.config.update({"checkpoints": ckpt_dir, "session_id": session_id})

hostname = platform.node()
print("host: ", hostname)
print("SLURM job ID: ", opt.jobid)
print("checkpoint directory: ", ckpt_dir)
print("all settings:\n")
print_options(opt)

implicit_function = networks.layers.CuboidWithImplicitFunGradient().apply

if opt.dataset == "nyu":
    trainset = NYURGBDataset(data_directory=opt.data_path, split='train', scale=1, split_mat=opt.nyu_split)
    valset = NYURGBDataset(data_directory=opt.data_path, split='val', scale=1, split_mat=opt.nyu_split)

    depth_mean = valset.depth_mean
    depth_stdd = valset.depth_stdd
    image_mean = torch.from_numpy(trainset.image_mean)
elif opt.dataset == "smh":
    trainset = SMHDataset(opt.data_path, split='train_random', scale=0.25, keep_in_mem=False)
    valset = SMHDataset(opt.data_path, split='val_random', scale=0.25, keep_in_mem=False)

    depth_mean = valset.depth_mean
    depth_stdd = valset.depth_stdd
    image_mean = torch.from_numpy(trainset.image_mean)
else:
    assert False, "unsupported dataset %s" % opt.dataset

trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=4, batch_size=opt.batch, drop_last=True)
valset_loader = torch.utils.data.DataLoader(valset, shuffle=False, num_workers=4, batch_size=opt.batch, drop_last=True)

devices = get_devices(opt)
fitting_device, consac_device, depth_device, inlier_device = devices

torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)

depth_model = get_depth_model(opt, devices)
consac = get_consac_model(opt, devices, train=True)

H, W, Y, H_, W_, Y_, M, P, S, Q, R, B, K, model_dim, data_dim, minimal_set_size, dimensions = \
    get_dimensions(opt, trainset)

inlier_function = consistency.soft_inlier_fun_gen(5. / opt.threshold, opt.threshold)
if not opt.no_occlusion_penalty:
    if opt.occlusion_penalty_schedule > 0:
        occlusion_crossover = opt.occlusion_penalty_schedule
    else:
        occlusion_crossover = opt.threshold
    outlier_function = consistency.soft_outlier_lin_fun_gen(5. / opt.threshold, opt.threshold, occlusion_crossover)
else:
    outlier_function = consistency.soft_outlier_fun_gen(5. / opt.threshold, opt.threshold)

if opt.minsolver == 'lbfgs':
    minimal_solver = CuboidFitLBFGS(a_max=opt.a_max, a_min=opt.a_min, norm_by_volume=True)
elif opt.minsolver == 'adam':
    minimal_solver = CuboidFitAdam(a_max=opt.a_max, a_min=opt.a_min, norm_by_volume=True, iterations=opt.fitting_iterations, lr=opt.fitting_lr, max_loss=1e-5)
else:
    minimal_solver = CuboidFitNN(lr=opt.solver_lr, a_max=opt.a_max, a_min=opt.a_min, device=fitting_device,
                                 load=opt.load_solver, layers=4, train=opt.train_solver,
                                 arch=opt.minsolver)

iteration = 0
first_epoch = opt.start_epoch

best_auc = 0
best_epoch = 0
last_auc = 0

for epoch in range(first_epoch, opt.epochs):

    print("Epoch ", epoch)

    if opt.occlusion_penalty_schedule > 0:
        occlusion_crossover = opt.occlusion_penalty_schedule - epoch*1./opt.epochs * (opt.occlusion_penalty_schedule-opt.threshold)
        outlier_function = consistency.soft_outlier_lin_fun_gen(5. / opt.threshold, opt.threshold, occlusion_crossover)

    if last_auc >= best_auc:
        best_auc = last_auc
        best_epoch = epoch
        is_best = True
    else:
        is_best = False

    print("best epoch: ", best_epoch)

    if opt.train_consac:
        save_consac_model(consac, ckpt_dir, epoch, is_best=is_best)
    if opt.train_depth:
        save_depth_model(depth_model, ckpt_dir, epoch, is_best=is_best)
    minimal_solver.save_checkpoint(ckpt_dir, epoch, is_best=is_best)

    modes = ['train'] if opt.noval else ['val', 'train']

    val_train_losses = []

    # alternate between validation and training passes:
    for mode in modes:

        loader = trainset_loader if mode == 'train' else valset_loader

        if mode == 'val':
            torch.set_grad_enabled(False)
            loader = valset_loader
            if not opt.seqransac:
                consac["model"].eval()
            set_eval_depth(depth_model)
        elif mode == 'train':
            torch.set_grad_enabled(True)
            loader = trainset_loader
            if opt.train_consac:
                consac["model"].train()
            if opt.train_depth:
                set_train_depth(depth_model)

        avg_losses_epoch = []
        avg_per_model_losses_epoch = [[] for _ in range(opt.instances)]
        avg_entropies = []
        mse_list = []
        weighted_mse_list = []
        batch_times = []
        oa_distances_for_val = []
        minsolver_residuals = []
        inlier_count_list = []
        auc_losses = []

        losses_minus_baseline_list = []
        baseline_list = []
        
        all_expected_rewards = []
        
        for idx, (image, intrinsic, true_coord_grid, labels, gt_models, gt_depth, _, mask) in enumerate(loader):

            inlier_counts_estm_best = np.zeros((P, M, K, B))
            inlier_counts_estm_expt = np.zeros((P, M, K, B))

            batch_start = time.time()

            if consac["optimizer"] is not None:
                consac["optimizer"].zero_grad()
            if depth_model["optimizer"] is not None:
                depth_model["optimizer"].zero_grad()

            states = torch.zeros((P, K, B, H, W, 1), device=consac_device)
            all_gradients = torch.zeros((P, M, K, B, Q, 1, Y_), device=consac_device)
            all_selected_gradients = torch.zeros((P, M, K, B, Q), device=consac_device)
            all_best_oa_distances_gt = None
            all_losses = torch.zeros((K, B), device=consac_device)
            all_losses_per_model = torch.zeros((M, K, B), device=consac_device)
            neg_inliers = torch.ones((P, M - 1, K, B, Y_), device=consac_device)
            all_best_inlier_counts_estm = None
            all_best_inlier_counts_gt = None
            all_inlier_counts_estm = torch.zeros((P, M, S, K, B), device=consac_device)
            all_inlier_counts_gt = torch.zeros((P, M, S, K, B), device=consac_device)
            all_best_single_hypos = torch.zeros((P, M, K, B), device=consac_device, dtype=torch.int64)

            all_log_probs = []
            all_log_q = []

            prev_inliers_estm = None
            prev_inliers_gt = None
            prev_distances_gt = None
            prev_distances_estm = None
            prev_occluded_distances_gt = None
            prev_occluded_distances_estm = None

            # get depth and generate coordinate grids (point clouds) using camera intrinsics:
            depth, depth_normalised, depth_mse, image_scaled = \
                estimate_depth(opt, image, image_mean, depth_model, dimensions, devices, depth_mean, depth_stdd, gt_depth)
            true_coord_grid = true_coord_grid.to(fitting_device)
            true_coord_grid, true_coord_grid_small, true_coord_flat, \
            estm_coord_grid, estm_coord_grid_small, estm_coord_flat, mask_flat, mask_small = \
                generate_coordinate_grids(depth, true_coord_grid, dimensions, devices, mask, gt=(opt.depth_model == 'gt'))

            # feature input for the CONSAC network:
            data = depth_normalised.to(states.device)

            residual_list = []

            for mi in range(opt.instances):

                # set the state input for the CONSAC network:
                if prev_inliers_estm is not None:
                    for pi in range(P):
                        for ki in range(K):
                            inliers_scaled = torch.nn.functional.interpolate(
                                prev_inliers_estm[pi, :, ki, :].view(B, 1, H_, W_), size=(H, W)).squeeze()
                            states[pi, ki, :, :, :, 0] = inliers_scaled

                # predict sampling weights:
                sampling_weight_maps, selection_weights, log_probs, log_q, entropy = \
                    estimate_sampling_weights(opt, dimensions, devices, data, states, consac,
                                              previous_inliers=None)

                sampling_weight_maps = sampling_weight_maps * mask_flat[None, None, None, :, None, None, :]

                all_log_probs += [log_probs]
                all_log_q += [log_q]

                avg_entropies += [entropy.detach().cpu().numpy().mean()]

                # sample and fit primitve hypotheses:
                models, choices, sel_choices, residual, _ = \
                    estimate_models(opt, dimensions, estm_coord_flat, sampling_weight_maps[:, :, :, :, :Q].detach(),
                                    selection_weights.detach(), minimal_solver)

                # set gradients needed for backprop:
                all_gradients[:, mi] = choices.sum(1).to(all_gradients.device)
                all_selected_gradients[:, mi] = sel_choices.sum(1).to(all_selected_gradients.device)

                residual_list += [residual]
                del choices, sel_choices

                # count inliers wrt estimated features for hypothesis selection:
                inliers_estm, distances_estm, occluded_distances_estm, oa_distances_estm = \
                    count_inliers(opt, models, estm_coord_flat, inlier_function, outlier_function, prev_distances_estm, prev_occluded_distances_estm,
                                  prev_inliers_estm, opt.inlier_mode_selection)

                # count inliers wrt ground truth for loss calculation:
                inliers_gt, distances_gt, occluded_distances_gt, oa_distances_gt = \
                    count_inliers(opt, models, true_coord_flat, inlier_function, outlier_function, prev_distances_gt,
                                  prev_occluded_distances_gt, prev_inliers_gt, opt.inlier_mode_loss)

                inliers_gt = inliers_gt * mask_flat[None, None, None, ...]
                inliers_estm = inliers_estm * mask_flat[None, None, None, ...]

                # select hypotheses and corresponding inliers:
                best_single_hypos = select_single_hypotheses(dimensions, inliers_estm)
                all_best_single_hypos[:, mi] = best_single_hypos

                prev_inliers_estm = torch.gather(inliers_estm, 1, best_single_hypos.view(P, 1, K, B, 1).expand(P, 1, K, B, Y_))
                prev_inliers_gt = torch.gather(inliers_gt, 1, best_single_hypos.view(P, 1, K, B, 1).expand(P, 1, K, B, Y_))
                prev_distances_gt = torch.gather(distances_gt, 1, best_single_hypos.view(P, 1, K, B, 1).expand(P, 1, K, B, Y_))
                prev_distances_estm = torch.gather(distances_estm, 1, best_single_hypos.view(P, 1, K, B, 1).expand(P, 1, K, B, Y_))
                prev_occluded_distances_gt = torch.gather(occluded_distances_gt, 1, best_single_hypos.view(P, 1, K, B, 1).expand(P, 1, K, B, Y_))
                prev_occluded_distances_estm = torch.gather(occluded_distances_estm, 1, best_single_hypos.view(P, 1, K, B, 1).expand(P, 1, K, B, Y_))

                prev_oa_distances_gt = torch.gather(oa_distances_gt, 1, best_single_hypos.view(P, 1, K, B, 1).expand(P, 1, K, B, Y_))

                if mi < M - 1:
                    neg_inliers[:, mi, :, :] = 1 - torch.clamp(prev_inliers_estm, min=0, max=1)

                all_inlier_counts_estm[:, mi] = inliers_estm.sum(-1).to(all_inlier_counts_estm.device)
                all_inlier_counts_gt[:, mi] = inliers_gt.sum(-1).to(all_inlier_counts_gt.device)

                all_best_inlier_counts_estm = \
                    torch.gather(all_inlier_counts_estm[:, mi], 1, best_single_hypos.view(P, 1, K, B).to(all_inlier_counts_estm.device)).squeeze(1)
                all_best_inlier_counts_gt = \
                                    torch.gather(all_inlier_counts_gt[:, mi], 1, best_single_hypos.view(P, 1, K, B).to(all_inlier_counts_estm.device)).squeeze(1)

            all_log_probs = torch.stack(all_log_probs, dim=1)
            all_log_q = torch.stack(all_log_q, dim=1)

            best_multi_hypos = torch.argmax(all_best_inlier_counts_estm, dim=0)

            inlier_count_list += [all_best_inlier_counts_estm.cpu().detach().numpy()]

            oa_distances_for_val += [np.sqrt(gather_oa_distances(best_multi_hypos, prev_oa_distances_gt.squeeze(1)))]

            mean_loss_per_model, expected_rewards = \
                compute_losses(opt, dimensions, all_inlier_counts_estm, all_inlier_counts_gt, all_best_single_hypos,
                               best_multi_hypos, all_losses, all_losses_per_model)

            all_expected_rewards += [expected_rewards.reshape([expected_rewards.shape[0], -1])]

            all_losses_per_model_estm = all_losses_per_model.detach()

            baselines_per_model = all_losses_per_model_estm.mean(dim=1)
            baselines_per_model = torch.clamp(baselines_per_model, min=-1, max=1)

            baseline_list += [baselines_per_model.detach().cpu().numpy()]

            losses_minus_baseline = all_losses_per_model_estm - baselines_per_model.unsqueeze(1)

            for mi in range(opt.instances):
                avg_per_model_losses_epoch[mi] += [baselines_per_model[mi].mean().detach().cpu().numpy().squeeze()]

            for ki in range(K):
                all_gradients[:, :, ki, :, :] *= losses_minus_baseline[:, ki, :].view(1, M, B, 1, 1, 1)
                all_selected_gradients[:, :, ki, :, :] *= losses_minus_baseline[:, ki, :].view(1, M, B, 1)

            losses_minus_baseline_list += [losses_minus_baseline.detach().cpu().numpy()]

            avg_loss = all_losses.mean()
            avg_losses_epoch += [avg_loss.detach().cpu().numpy()]

            mean_residual = sum(residual_list) * 1. / len(residual_list)
            minsolver_residuals += [mean_residual.cpu().detach().numpy()]

            if mode == 'train':
                if not opt.no_backward:
                    backward_pass(opt, consac["optimizer"], depth_model["optimizer"], mean_loss_per_model,
                                  neg_inliers, all_gradients, all_selected_gradients, depth_mse, dimensions, devices,
                                  minimal_solver, mean_residual, all_log_probs, all_log_q)
                else:
                    consac["optimizer"].zero_grad()
                    depth_model["optimizer"].zero_grad()

                iteration += 1

            mse_list += [depth_mse.detach().cpu().numpy()]

            batch_end = time.time()
            batch_times += [batch_end-batch_start]

            format_string = "(%s) batch %6d / %d : %.4f -- time: %d (%d)"
            value_list = [mode, idx + 1, len(loader), avg_loss.item(), int(batch_times[-1] * 1000),
                          int(np.mean(batch_times) * 1000)]
            print(format_string % tuple(value_list), end="\n")

            del data, baselines_per_model, avg_loss, depth, true_coord_grid, all_inlier_counts_gt, \
                true_coord_grid_small, estm_coord_grid, estm_coord_grid_small, all_gradients, all_selected_gradients, \
                all_losses, all_losses_per_model, neg_inliers


        log_data = {}

        if mode == 'val':
            auc_values, mean_oa = calc_auc_values(oa_distances_for_val)
            log_data['%s/%s' % (mode, "mean_oa")] = mean_oa
            last_auc = auc_values["auc_at_20"]
            print("mean OA %.3f" % mean_oa)
            for key in auc_values:
                print("AUC at %s: %.3f" % (key, auc_values[key]))
                log_data['%s/%s' % (mode, key)] = auc_values[key]

        avg_loss_epoch = sum([l for l in avg_losses_epoch]) / len(avg_losses_epoch)
        avg_entropy_epoch = sum([l for l in avg_entropies]) / len(avg_entropies)
        print("(%s) Avg epoch loss: %.3f" % (mode, avg_loss_epoch))
        print("(%s) Avg epoch entropy: %.3f" % (mode, avg_entropy_epoch))
        log_data['%s/loss_epoch_avg' % mode] = avg_loss_epoch
        log_data['%s/entropy' % mode] = avg_entropy_epoch
        log_data['%s/minsolver_residual' % mode] = np.mean(minsolver_residuals)
        log_data['%s/inlier_count_avg' % mode] = np.mean(inlier_count_list)


        baselines = np.concatenate(baseline_list, axis=-1)
        baseline_mean = np.mean(baselines, axis=-1)
        baseline_var = np.var(baselines, axis=-1)

        all_expected_rewards = np.concatenate(all_expected_rewards, axis=-1)
        expected_rewards_per_model = np.mean(all_expected_rewards, axis=-1)

        for mi in range(opt.instances):
            avg_per_model_loss_epoch = sum([l for l in avg_per_model_losses_epoch[mi]]) / len(avg_per_model_losses_epoch[mi])
            log_data['%s/model_loss_avg/%d' % (mode, mi)] = avg_per_model_loss_epoch
            log_data['%s/baseline_mean/%d' % (mode, mi)] = baseline_mean[mi]
            log_data['%s/baseline_var/%d' % (mode, mi)] = baseline_var[mi]
            log_data['%s/expected_inliers/%d' % (mode, mi)] = expected_rewards_per_model[mi]

        losses_var = np.var(losses_minus_baseline_list)
        log_data['%s/mean_loss' % mode] = mean_loss_per_model.item()
        log_data['%s/depth_mse' % mode] = np.mean(mse_list)

        val_train_losses += [avg_loss_epoch]

        wandb.log(log_data, step=iteration)

    loss_log_writer.writerow([epoch] + val_train_losses)

if last_auc >= best_auc:
    best_auc = last_auc
    best_epoch = opt.epochs
    is_best = True
else:
    is_best = False

print("Best Epoch: %d" % best_epoch)

if opt.train_consac:
    save_consac_model(consac, ckpt_dir, opt.epochs, is_best=is_best)
if opt.train_depth:
    save_depth_model(depth_model, ckpt_dir, opt.epochs, is_best=is_best)
minimal_solver.save_checkpoint(ckpt_dir, opt.epochs, is_best=is_best)

loss_log.close()
