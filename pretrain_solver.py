from datasets.synthetic_cuboids import SynthCuboidsDataset
from torch.utils.data import Dataset
import glob
from util.misc import *
from util.fitting import *
import time
import random

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

if __name__ == "__main__":

    import csv
    import argparse

    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--layers', default=4, type=int, help='number of transformer layers')
    parser.add_argument('--epochs', default=300, type=int, help='number of training epochs')
    parser.add_argument('--epoch_size', default=4096*500, type=int, help='number of samples per epoch')
    parser.add_argument('--batch', default=4096, type=int, help='batch size')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--mss', default=6, type=int, help='minimal set size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--ckpt_dir', default="./tmp/checkpoints/cuboid_solver", help='destination folder network checkpoints')
    parser.add_argument('--dataset', default="nyu", help='use dataset-specific settings: nyu or smh')
    opt = parser.parse_args()

    epochs = opt.epochs
    batch_size = opt.batch

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)

    ckpt_dir = opt.ckpt_dir

    if os.path.isdir(ckpt_dir):
        ckpt_dirs = glob.glob(os.path.join(opt.ckpt_dir, "session_*"))
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

    ckpt_dir_exists = True
    while ckpt_dir_exists:
        ckpt_dir = os.path.join(opt.ckpt_dir, "session_%03d_rb%d_lr%f_b%d_%d" % (session_id, opt.layers, opt.lr, opt.batch, opt.seed))
        time.sleep(np.random.uniform(0.5, 3.0))
        if os.path.exists(ckpt_dir):
            session_id += 1
        else:
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_dir_exists = False

    log_file = os.path.join(ckpt_dir, "output.log")
    log = Tee(log_file, "w", file_only=False)

    print(opt)
    print("saving to: ", ckpt_dir)

    loss_log_file = os.path.join(ckpt_dir, "loss.log")
    loss_log = open(loss_log_file, mode='w')
    loss_log_writer = csv.writer(loss_log, delimiter=',')
    loss_log_writer.writerow(['epoch', 'val_loss', 'train_loss'])

    if opt.dataset == "smh":
        dataset = SynthCuboidsDataset(opt.epoch_size, mss=opt.mss, size_range=(2.0, 30.0), dist_range=(1.0, 50.0), xy_range=(-50, 50))
    elif opt.dataset == "nyu":
        dataset = SynthCuboidsDataset(opt.epoch_size, mss=opt.mss, size_range=(0.01, 2.0), dist_range=(0.5, 10.0), xy_range=(-5, 5))
    else:
        assert False

    loader = torch.utils.data.DataLoader(dataset, shuffle=True, num_workers=14, batch_size=batch_size, drop_last=True,
                                         worker_init_fn=worker_init_fn)

    device = torch.device('cuda', 0)

    primitive_solver = CuboidFitNN(lr=opt.lr, a_max=30.0, a_min=2.0, device=device, layers=opt.layers)

    torch.set_grad_enabled(True)

    primitive_solver.train()

    iteration = 0
    best_loss = np.inf
    best_epoch = 0
    for epoch in range(epochs):
        print("epoch: ", epoch)

        log_data = {}

        np.random.seed(epoch)

        avg_losses_epoch = []

        for idx, (size, axis_angle, t, samples) in enumerate(loader):

            optim_params, mean_residual, _ = primitive_solver.fit(samples, None)

            loss = mean_residual

            avg_losses_epoch += [loss.cpu().detach().numpy()]

            primitive_solver.zero_grad()

            loss.backward()

            params = list(primitive_solver.network.parameters())

            primitive_solver.backward_step()

            format_string = "batch %6d / %d : %.4f "
            value_list = [idx + 1, len(loader), loss.item()]
            print(format_string % tuple(value_list), end="\n")

            iteration += 1

        avg_loss_epoch = sum([l for l in avg_losses_epoch]) / len(avg_losses_epoch)
        print("\nAvg epoch loss: %.3f" % (avg_loss_epoch))
        log_data["loss_epoch_avg"] = avg_loss_epoch

        primitive_solver.save_checkpoint(ckpt_dir, epoch)

        if avg_loss_epoch < best_loss:
            best_loss = avg_loss_epoch
            best_epoch = epoch
        elif avg_loss_epoch > 2*best_loss:
            primitive_solver.load_checkpoint(ckpt_dir, best_epoch)
            print("load ckpt from epoch ", best_epoch)

    primitive_solver.save_checkpoint(ckpt_dir, epochs)
