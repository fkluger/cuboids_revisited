from util.train import *
from util.options import *
from util.forward import *
from util.fitting import *
from util.initialisation import *
from datasets.nyu_depth.nyu_depth import NYURGBDataset
import networks.layers
import numpy as np
import platform
import random
import torch
import sys
import time

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

opt = get_options()

gettrace = getattr(sys, 'gettrace', None)
if gettrace():
    opt.debugging = True

ckpt_dir, log, loss_log_writer, loss_log, tensorboard_writer = get_log_and_checkpoint_directory(opt)

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
else:
    assert False, "unsupported dataset %s" % opt.dataset

trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=1, batch_size=opt.batch, drop_last=True)
valset_loader = torch.utils.data.DataLoader(valset, shuffle=False, num_workers=1, batch_size=opt.batch, drop_last=True)

devices = get_devices(opt)
fitting_device, consac_device, depth_device, inlier_device = devices

torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)

depth_model = get_depth_model(opt, devices)
consac = get_consac_model(opt, devices)

H, W, Y, H_, W_, Y_, M, P, S, Q, R, B, K, model_dim, data_dim, minimal_set_size, dimensions = \
    get_dimensions(opt, trainset)

inlier_function = consistency.soft_inlier_fun_gen(5. / opt.threshold, opt.threshold)

if opt.lbfgs:
    minimal_solver = CuboidFitLBFGS(a_max=opt.a_max, norm_by_volume=True)
else:
    minimal_solver = CuboidFitAdam(a_max=opt.a_max, norm_by_volume=True)

iteration = 0
first_epoch = 0

best_auc = 0
best_epoch = 0
last_auc = 0

for epoch in range(first_epoch, opt.epochs):

    print("Epoch ", epoch)

    if epoch % opt.eval_freq == 0 or epoch == opt.epochs-1:

        if last_auc >= best_auc:
            best_auc = last_auc
            best_epoch = epoch
            is_best = True
        else:
            is_best = False

        if opt.train_consac:
            save_consac_model(consac, ckpt_dir, epoch, is_best=is_best)
        if opt.train_depth or opt.train_mse:
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

        for idx, (image, intrinsic, true_coord_grid, labels, gt_models, gt_depth, _) in enumerate(loader):

            batch_start = time.time()

            if consac["optimizer"] is not None:
                consac["optimizer"].zero_grad()
            if depth_model["optimizer"] is not None:
                depth_model["optimizer"].zero_grad()

            states = torch.zeros((P, M, K, B, H, W, 1), device=consac_device)
            all_gradients = torch.zeros((P, M, K, B, Q, 1, Y_), device=consac_device)
            all_selected_gradients = torch.zeros((P, M, K, B, Q), device=consac_device)
            all_best_oa_distances_gt = None
            all_losses = torch.zeros((K, B), device=consac_device)
            all_losses_per_model = torch.zeros((M, K, B), device=consac_device)
            neg_inliers = torch.ones((P, M + 1, K, B, Y_), device=consac_device)
            all_best_inlier_counts_estm = torch.zeros((P, M, K, B), device=depth_device)
            all_inlier_counts_estm = torch.zeros((P, M, S, K, B), device=depth_device)
            all_inlier_counts_gt = torch.zeros((P, M, S, K, B), device=depth_device)
            mean_oa_distances_gt = torch.zeros((P, M, S, K, B), device=depth_device)

            prev_inliers_estm = None
            prev_inliers_gt = None
            prev_distances_gt = None
            prev_occluded_distances_gt = None

            # get depth and generate coordinate grids (point clouds) using camera intrinsics:
            depth, depth_normalised, depth_mse = \
                estimate_depth(opt, image, image_mean, depth_model, dimensions, devices, depth_mean, depth_stdd, gt_depth)
            true_coord_grid = true_coord_grid.to(fitting_device)
            true_coord_grid, true_coord_grid_small, true_coord_flat, \
            estm_coord_grid, estm_coord_grid_small, estm_coord_flat = \
                generate_coordinate_grids(depth, true_coord_grid, dimensions, devices)

            # feature input for the CONSAC network:
            data = depth_normalised.to(states.device)

            for mi in range(opt.instances):

                torch.set_grad_enabled(False)

                # set the state input for the CONSAC network:
                if prev_inliers_estm is not None:
                    for pi in range(P):
                        for ki in range(K):
                            inliers_scaled = torch.nn.functional.interpolate(
                                prev_inliers_estm[pi, :, ki, :].view(B, 1, H_, W_), size=(H, W)).squeeze()
                            states[pi, mi, ki, :, :, :, 0] = inliers_scaled

                # predict sampling weights:
                sampling_weight_maps, selection_weights, log_probs, log_q, entropy = \
                    estimate_sampling_weights(opt, dimensions, devices, data, states[:, mi], consac,
                                              previous_inliers=None)

                avg_entropies += [entropy.detach().cpu().numpy().mean()]

                if mode == 'train':
                    torch.set_grad_enabled(True)

                # sample and fit primitve hypotheses:
                models, choices, sel_choices, residual = \
                    estimate_models(opt, dimensions, estm_coord_flat, sampling_weight_maps[:, :, :, :, :Q].detach(),
                                    selection_weights.detach(), minimal_solver)

                # set gradients needed for backprop:
                all_gradients[:, mi] = choices.sum(1).to(all_gradients.device)
                all_selected_gradients[:, mi] = sel_choices.sum(1).to(all_selected_gradients.device)
                minsolver_residuals += [residual.cpu().detach().numpy()]
                del choices, sel_choices, residual

                # count inliers wrt estimated features for hypothesis selection:
                inliers_estm, distances_estm, occluded_distances_estm = \
                    count_inliers(opt, models, estm_coord_flat, inlier_function, None, None, prev_inliers_estm,
                                  occlusion_aware=(not opt.no_oai_sampling))
                del distances_estm, occluded_distances_estm

                # count inliers wrt ground truth for loss calculation:
                inliers_gt, distances_gt, occluded_distances_gt = \
                    count_inliers(opt, models, true_coord_flat, inlier_function, prev_distances_gt,
                                  prev_occluded_distances_gt, prev_inliers_gt, occlusion_aware=(not opt.no_oai_loss))

                # select hypotheses and corresponding inliers:
                best_single_hypos, _ = select_single_hypotheses(opt, dimensions, inliers_estm)
                del _
                best_inliers_estm = torch.gather(inliers_estm, 1,
                                               best_single_hypos.view(P, 1, K, B, 1).expand(P, 1, K, B, Y_))
                neg_inliers[:, mi + 1, :, :] = 1 - best_inliers_estm
                prev_inliers_estm = best_inliers_estm
                prev_inliers_gt = torch.gather(inliers_gt, 1, best_single_hypos.view(P, 1, K, B, 1).expand(P, 1, K, B, Y_))
                prev_distances_gt = torch.gather(distances_gt, 1, best_single_hypos.view(P, 1, K, B, 1).expand(P, 1, K, B, Y_))
                prev_occluded_distances_gt = torch.gather(occluded_distances_gt, 1, best_single_hypos.view(P, 1, K, B, 1).expand(P, 1, K, B, Y_))

                oa_distances_gt = torch.max(distances_gt, occluded_distances_gt)

                all_inlier_counts_estm[:, mi] = inliers_estm.sum(-1).to(all_inlier_counts_estm.device)
                all_inlier_counts_gt[:, mi] = inliers_gt.sum(-1).to(all_inlier_counts_gt.device)
                mean_oa_distances_gt[:, mi] = oa_distances_gt.to(mean_oa_distances_gt.device).mean(dim=-1)

                all_best_oa_distances_gt = \
                    torch.gather(oa_distances_gt, 1,
                                 best_single_hypos.view(P, 1, K, B, 1).expand(P, 1, K, B, Y_).to(
                                     oa_distances_gt.device)).squeeze(1)

                all_best_inlier_counts_estm[:, mi] = \
                    torch.gather(all_inlier_counts_estm[:, mi], 1, best_single_hypos.view(P, 1, K, B).to(all_inlier_counts_estm.device)).squeeze(1)

                if mode == 'train':
                    torch.set_grad_enabled(True)

            final_inlier_counts = all_best_inlier_counts_estm[:, -1]
            best_multi_hypos = torch.argmax(final_inlier_counts, dim=0)

            oa_distances_for_val += [np.sqrt(gather_oa_distances(best_multi_hypos, all_best_oa_distances_gt))]

            all_losses_per_model_list = \
                compute_losses(opt, dimensions, all_inlier_counts_estm, all_inlier_counts_gt, mean_oa_distances_gt,
                               best_multi_hypos, all_losses, all_losses_per_model)

            all_losses_per_model_estm = all_losses_per_model.detach()

            baselines_per_model = all_losses_per_model_estm.mean(dim=1)
            baselines = all_losses.mean(dim=0)

            for mi in range(opt.instances):
                avg_per_model_losses_epoch[mi] += [baselines_per_model[mi].mean().detach().cpu().numpy().squeeze()]

            for bi in range(0, data.size(0)):
                baseline = baselines[bi]
                for ki in range(K):
                    all_gradients[:, :, ki, bi, :] *= (all_losses_per_model_estm[:, ki, bi].view(1, M, 1, 1, 1) -
                                                   baselines_per_model[:, bi].view(1, M, 1, 1, 1))
                    all_selected_gradients[:, :, ki, bi, :] *= (all_losses_per_model_estm[:, ki, bi].view(1, M, 1) -
                                                   baselines_per_model[:, bi].view(1, M, 1))

            avg_loss = all_losses.mean()
            avg_losses_epoch += [avg_loss.detach().cpu().numpy()]

            if mode == 'train':
                if not opt.no_backward:
                    backward_pass(opt, consac["model"], consac["optimizer"], depth_model["optimizer"],
                                  all_losses_per_model_list, data, states, neg_inliers, all_gradients,
                                  all_selected_gradients, depth_mse, dimensions, devices, minimal_solver)
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

            del data, baselines_per_model, baselines, avg_loss, depth, true_coord_grid, all_inlier_counts_gt, \
                true_coord_grid_small, estm_coord_grid, estm_coord_grid_small, all_gradients, all_selected_gradients, \
                all_losses, all_losses_per_model, neg_inliers

            for loss in all_losses_per_model_list:
                del loss

        if mode == 'val':
            auc_values = calc_auc_values(oa_distances_for_val)
            for key in auc_values:
                print("AUC at %s: %.3f" % (key, auc_values[key]))
                tensorboard_writer.add_scalar('%s/%s' % (mode, key), auc_values[key], iteration)

        avg_loss_epoch = sum([l for l in avg_losses_epoch]) / len(avg_losses_epoch)
        avg_entropy_epoch = sum([l for l in avg_entropies]) / len(avg_entropies)
        print("(%s) Avg epoch loss: %.3f" % (mode, avg_loss_epoch))
        print("(%s) Avg epoch entropy: %.3f" % (mode, avg_entropy_epoch))
        tensorboard_writer.add_scalar('%s/loss_epoch_avg' % mode, avg_loss_epoch, iteration)
        tensorboard_writer.add_scalar('%s/entropy' % mode, avg_entropy_epoch, iteration)
        tensorboard_writer.add_scalar('%s/minsolver_residual' % mode, np.mean(minsolver_residuals), iteration)

        for mi in range(opt.instances):
            avg_per_model_loss_epoch = sum([l for l in avg_per_model_losses_epoch[mi]]) / len(avg_per_model_losses_epoch[mi])
            tensorboard_writer.add_scalar('%s/model_loss_avg_%d' % (mode, mi), avg_per_model_loss_epoch, iteration)

        tensorboard_writer.add_scalar('%s/depth_mse' % mode, np.mean(mse_list), iteration)

        val_train_losses += [avg_loss_epoch]

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
