import argparse


def get_options():
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # train and eval:
    parser.add_argument('--data_path', default="/data/kluger/datasets/nyu_files", help='path to dataset')
    parser.add_argument('--nyu_split', default="./bts/utils/splits.mat", help='path to NYU dataset split')
    parser.add_argument('--dataset', default="nyu", help='name of dataset to use')
    parser.add_argument('--depth_model', default="gt", help='name of feature extractor (bts, gt)')
    parser.add_argument('--threshold', '-t', type=float, default=0.004, help='tau - inlier threshold')
    parser.add_argument('--hyps', type=int, default=4096, help='S - inner hypotheses (single instance hypotheses) ')
    parser.add_argument('--outerhyps', type=int, default=1, help='P - outer hypotheses (multi-hypotheses)')
    parser.add_argument('--instances', type=int, default=6, help='M - max. number of instances')
    parser.add_argument('--num_probs', default=4, type=int, help='Q - number of sampling weight maps')
    parser.add_argument('--min_prob', type=float, default=1e-8,
                        help='min sampling weight to avoid degenerate distributions')
    parser.add_argument('--mss', default=9, type=int, help='GPU ID to use')
    parser.add_argument('--depth_gpu', default="-1",  help='GPU ID for the depth estimator (-1 for CPU)')
    parser.add_argument('--consac_gpu', default="-1",  help='GPU ID for the sampling weight estimator (-1 for CPU)')
    parser.add_argument('--inlier_gpu', default="-1",  help='GPU ID for inlier computation (-1 for CPU)')
    parser.add_argument('--fitting_gpu', default="-1",  help='GPU ID for cuboid parameter fitting (-1 for CPU)')
    parser.add_argument('--load', default="./models/consac_weights.net", type=str,
                        help='sampling weight estimator: load pretrained NN weights from file')
    parser.add_argument('--load_depth', default="./models/bts_weights.net", type=str,
                        help='depth estimator: load pretrained NN weights from file')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--jobid', default=-1, type=int, help='SLURM job ID (for logging/debugging)')
    parser.add_argument('--fitting_iterations', default=50, type=int, help='cuboid fitting iterations')
    parser.add_argument('--a_min', type=float, default=0.001, help='min. cuboid size (smallest side)')
    parser.add_argument('--a_max', type=float, default=2., help='max. cuboid size (longest side)')
    parser.add_argument('--normalise_depth', dest='normalise_depth', action='store_true',
                        help='normalise depth using dataset mean and std')
    parser.add_argument('--bn_on_input', dest='bn_on_input', action='store_true',
                        help='use batchnorm layer to normalise depth')
    parser.add_argument('--lbfgs', dest='lbfgs', action='store_true', help='use LBFGS instead of Adam for cuboid fitting')
    parser.add_argument('--seqransac', dest='seqransac', action='store_true',
                        help='Use Sequential RANSAC instead of CONSAC for sampling')

    parser.add_argument('--no_oai_sampling', dest='no_oai_sampling', action='store_true', help='')
    parser.add_argument('--no_oai_loss', dest='no_oai_loss', action='store_true', help='')
    parser.add_argument('--align_depth', dest='align_depth', action='store_true', help='')

    # training only:
    parser.add_argument('--batch', '-bs', type=int, default=2, help='B - batch size')
    parser.add_argument('--samplecount', '-ss', type=int, default=2, help='K - sample count')
    parser.add_argument('--ckpt_dir', default='./results/train',
                        help='directory for storing NN weight checkpoints')
    parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
    parser.add_argument('--eval_freq', type=int, default=1, help='eval on validation set every n epochs')
    parser.add_argument('--val_iter', type=int, default=1, help='number of eval runs on validation set')
    parser.add_argument('--consac_lr', type=float, default=1e-5, help='learning rate for sampling weight NN')
    parser.add_argument('--depth_lr', type=float, default=1e-9, help='learning rate for depth estimation NN')
    parser.add_argument('--loss_clamp', type=float, default=0.3, help='clamp absolute value of losses')
    parser.add_argument('--max_prob_loss', type=float, default=0.01, help='kappa_im - inlier masking regularisation')
    parser.add_argument('--train_depth', dest='train_depth', action='store_true', help='train the depth estimator')
    parser.add_argument('--train_mse', dest='train_mse', action='store_true', help='train the depth estimator using MSE loss')
    parser.add_argument('--train_consac', dest='train_consac', action='store_true', help='train the sampling weight estimator')
    parser.add_argument('--minimise_corr', type=float, default=1., help='kappa_corr - correlation min. regularisation')
    parser.add_argument('--maximise_second_entropy', type=float, default=1., help='kappa_entropy - entropy max. regularisation')
    # debug:
    parser.add_argument('--no_backward', dest='no_backward', action='store_true', help='disable backward pass')
    parser.add_argument('--noval', dest='noval', action='store_true', help='skip validation phase')
    parser.add_argument('--debugging', dest='debugging', action='store_true', help='')

    # evaluation only:
    parser.add_argument('--eval_results', default="./results/eval", help='path to save results to')
    parser.add_argument('--depth_cache', default="./tmp/depth_cache", help='destination folder for depth cache')
    parser.add_argument('--read_cache', dest='read_cache', action='store_true', help='read depth from cache')
    parser.add_argument('--write_cache', dest='write_cache', action='store_true', help='write depth to cache')
    parser.add_argument('--write_cache_only', dest='write_cache_only', action='store_true', help='only write depth to cache')
    parser.add_argument('--visualise', dest='visualise', action='store_true', help='plot results for each image')
    parser.add_argument('--save_all', dest='save_all', action='store_true', help='')
    parser.add_argument('--runcount', default=1, type=int, help='number of runs for each image')
    parser.add_argument('--inlier_cutoff', default=10, type=int, help='Theta - min. number of inliers before termination')
    parser.add_argument('--sampleid', default=0, type=int, help='start at image index')
    parser.add_argument('--split', default="test", help='dataset split to use')

    return parser.parse_args()


def print_options(opt):
    keys = list(vars(opt).keys())
    keys.sort()
    for arg in keys:
        print(arg, ":", getattr(opt, arg))
