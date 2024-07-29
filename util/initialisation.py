from networks.res_net import Network, SmallNetwork
from bts.pytorch.bts import BtsModel
from util.tee import Tee
from torch import nn
import torch.optim as optim
from collections import namedtuple
import torch
import os
import glob
import json
import csv
import copy


def load_opts_for_eval(opt):

    override_options = ["a_max", "a_min", "bn_on_input", "correct_oai", "cuboidfitnn",
                        "cuboidfitnnandadam", "cuboids", "depth_model", "fit_smallest", "fitting_iterations", "lbfgs",
                        "min_prob", "mss", "normalise_depth", "num_probs", "seqransac",
                        "spare_prob", "threshold", "unconditional", "uniform"]

    if os.path.isdir(opt.load):
        args_file = os.path.join(opt.load, "commandline_args.txt")

        with open(args_file) as f:
            orig_args = json.load(f)

        for orig_key, orig_value in orig_args.items():
            if orig_key in override_options:
                opt.__dict__[orig_key] = orig_value

        consac_path = os.path.join(opt.load, 'consac_weights_%06d.net' % opt.epochs)
        depth_path = os.path.join(opt.load, 'depth_weights_%06d.net' % opt.epochs)

        if os.path.exists(consac_path):
            opt.load = consac_path
        if os.path.exists(depth_path):
            opt.load_depth = depth_path

    return opt


def load_opts_for_resume(opt):

    new_opt = copy.deepcopy(opt)

    if os.path.isdir(opt.resume):
        print("resume training from ", opt.resume)
        args_file = os.path.join(opt.resume, "commandline_args.txt")

        with open(args_file) as f:
            orig_args = json.load(f)

        for orig_key, orig_value in orig_args.items():
            if orig_key in new_opt.__dict__.keys():
                new_opt.__dict__[orig_key] = orig_value

        consac_path = os.path.join(opt.resume, 'consac_weights_%06d.net' % new_opt.epochs)
        optimizer_path = os.path.join(opt.resume, 'consac_optimizer_%06d.net' % new_opt.epochs)
        depth_path = os.path.join(opt.resume, 'depth_weights_%06d.net' % new_opt.epochs)

        new_opt.start_epoch = new_opt.epochs
        new_opt.epochs = opt.epochs + new_opt.start_epoch

        new_opt.debugging = opt.debugging
        new_opt.noval = opt.noval
        new_opt.jobid = opt.jobid
        # new_opt.notrain = opt.notrain

        if os.path.exists(consac_path):
            new_opt.load = consac_path
            new_opt.finetune = True
        if os.path.exists(optimizer_path):
            new_opt.load_optimizer = optimizer_path
        if os.path.exists(depth_path):
            new_opt.load_depth = depth_path

    return new_opt



def get_dimensions(opt, dataset):
    image_size = dataset.get_image_size()

    minimal_set_size = opt.mss

    H = image_size[0]
    W = image_size[1]
    Y = H * W

    H_ = H // 8
    W_ = W // 8
    Y_ = W_ * H_

    M = opt.instances
    P = opt.outerhyps
    S = opt.hyps
    Q = opt.num_probs
    K = opt.samplecount

    R = 1

    B = opt.batch
    model_dim = 9

    data_dim = 2
    dimensions = {"M": M, "P": P, "S": S, "K": K, "Q": Q, "R": R, "B": B, "H": H, "W": W, "Y": Y, "H_": H_, "W_": W_,
                  "Y_": Y_, "data": data_dim, "mss": minimal_set_size, "model": model_dim}

    return H, W, Y, H_, W_, Y_, M, P, S, Q, R, B, K, model_dim, data_dim, minimal_set_size, dimensions


def get_log_and_checkpoint_directory(opt):

    parent_ckpt_dir = os.path.join(opt.ckpt_dir, opt.wandb_group)
    os.makedirs(parent_ckpt_dir, exist_ok=True)

    dir_success = False

    while not dir_success:
        if os.path.isdir(opt.ckpt_dir):
            ckpt_dirs = glob.glob(os.path.join(opt.ckpt_dir, opt.wandb_group, "session_*"))
            ckpt_dirs.sort()
            if len(ckpt_dirs) > 0:
                last_ckpt_dir = os.path.split(ckpt_dirs[-1])[1]
                try:
                    last_session_id = int(last_ckpt_dir[8:11])
                    session_id = last_session_id + 1
                except:
                    session_id = 0
            else:
                session_id = 0
        else:
            session_id = 0
        if opt.debugging:
            ckpt_dir = os.path.join(opt.ckpt_dir, opt.wandb_group, "debug_session")
            os.makedirs(ckpt_dir, exist_ok=True)
            dir_success = True
        else:
            ckpt_dir = os.path.join(opt.ckpt_dir, opt.wandb_group, "session_%03d_%s_%s" %
                                    (session_id, opt.depth_model, opt.minsolver))
            try:
                os.makedirs(ckpt_dir, exist_ok=False)
                dir_success = True
            except FileExistsError as err:
                print(err)
                print("%s exists, try again.." % ckpt_dir)

    log_file = os.path.join(ckpt_dir, "output.log")
    print("log file: ", log_file)
    log = Tee(log_file, "w", file_only=False)

    loss_log_file = os.path.join(ckpt_dir, "loss.log")
    loss_log = open(loss_log_file, mode='w')
    loss_log_writer = csv.writer(loss_log, delimiter=',')
    loss_log_writer.writerow(['epoch', 'val_loss', 'train_loss'])

    with open(os.path.join(ckpt_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(opt.__dict__, f, indent=2)
    #
    # tensorboard_directory = ckpt_dir + "/tensorboard/"
    # if not os.path.exists(tensorboard_directory):
    #     os.makedirs(tensorboard_directory)
    # tensorboard_writer = SummaryWriter(tensorboard_directory)

    return ckpt_dir, log, loss_log_writer, loss_log, session_id#, tensorboard_writer


def get_devices(opt):
    depth_device_ids = [int(x) for x in opt.depth_gpu.split(",")]
    if depth_device_ids[0] is None or depth_device_ids[0] < 0 or not torch.cuda.is_available():
        depth_device = torch.device('cpu')
    else:
        depth_device = torch.device('cuda', depth_device_ids[0])

    if opt.consac_gpu is None or int(opt.consac_gpu) < 0 or not torch.cuda.is_available():
        consac_device = torch.device('cpu')
    else:
        consac_device = torch.device('cuda', int(opt.consac_gpu))

    if opt.fitting_gpu is None or int(opt.fitting_gpu) < 0 or not torch.cuda.is_available():
        fitting_device = torch.device('cpu')
    else:
        fitting_device = torch.device('cuda', int(opt.fitting_gpu))

    if opt.inlier_gpu is None or int(opt.inlier_gpu) < 0 or not torch.cuda.is_available():
        inlier_device = torch.device('cpu')
    else:
        inlier_device = torch.device('cuda', int(opt.inlier_gpu))

    return fitting_device, consac_device, depth_device, inlier_device


def get_depth_model(opt, devices):

    depth_device = devices[2]

    if opt.depth_model == "bts":

        depth_device_ids = [int(x) for x in opt.depth_gpu.split(",")]

        BtsArgs = namedtuple('BtsArgs', ['encoder', 'bts_size', 'max_depth',  'dataset', 'pretrained'])
        args = BtsArgs(encoder='densenet161_bts', bts_size=512, max_depth=10, dataset='nyu', pretrained=False)

        model = BtsModel(params=args, bn_on_final_depth=True)

        loaded_dict = torch.load(opt.load_depth, map_location=depth_device)

        model = nn.DataParallel(model, device_ids=depth_device_ids)

        model.to(depth_device)
        if "model" in loaded_dict.keys():
            model.load_state_dict(loaded_dict["model"], strict=False)
        else:
            model.load_state_dict(loaded_dict, strict=False)

        feature_optimizer = optim.Adam(model.parameters(), lr=opt.depth_lr, eps=1e-4, weight_decay=1e-4)

        return {"name": opt.depth_model, "model": model,
                "optimizer": feature_optimizer, "height": 480, "width": 640}

    elif opt.depth_model == "dummy":

        model = torch.ones((opt.batch, 480, 640, 1), dtype=torch.float32, device=depth_device, requires_grad=True)

        feature_optimizer = optim.Adam([model], lr=opt.depth_lr, eps=1e-4, weight_decay=1e-4)

        return {"name": opt.depth_model, "model": model,
                "optimizer": feature_optimizer, "height": 480, "width": 640}

    elif opt.depth_model == "gt":
        return {"name": opt.depth_model, "model": None,
                "optimizer": None, "height": 480, "width": 640}

    else:
        assert False, "unknown depth model: %s" % opt.depth_model


def get_consac_model(opt, devices, train=False):

    if opt.seqransac:
        return {"model": None, "optimizer": None}

    consac_device = devices[1]

    net_fun = Network

    data_dim = 2

    consac_model = net_fun(data_channels=data_dim, instance_norm=True, feature_size=0, bn_on_input=False,
                           num_probs=opt.num_probs, separate_probs=1,
                           additional_prob=False)
    if opt.load is not None and (opt.finetune or not train) and len(opt.load) > 0:
        print("load consac weights: ", opt.load)
        consac_model.load_state_dict(torch.load(opt.load, map_location=consac_device), strict=True)
    consac_model = consac_model.to(consac_device)

    consac_optimizer = optim.Adam(consac_model.parameters(), lr=opt.consac_lr, eps=1e-4, weight_decay=1e-4)
    if opt.load_optimizer is not None:
        consac_optimizer.load_state_dict(torch.load(opt.load_optimizer, map_location=consac_device))

    return {"model": consac_model, "optimizer": consac_optimizer, "scale": 1./8}
