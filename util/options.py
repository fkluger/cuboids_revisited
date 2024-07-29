import argparse


def get_options():
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # train and eval:
    ## dataset:
    parser.add_argument('--data_path', default="/data/kluger/datasets/nyu_files", help='path to dataset')
    parser.add_argument('--nyu_split', default="./bts/utils/splits.mat", help='path to NYU dataset split')
    parser.add_argument('--dataset', default="nyu", help='name of dataset to use')
    ## hyperparameters:
    parser.add_argument('--threshold', '-t', type=float, default=0.004, help='tau - inlier threshold')
    parser.add_argument('--hyps', type=int, default=4096, help='S - inner hypotheses (single instance hypotheses) ')
    parser.add_argument('--outerhyps', type=int, default=1, help='P - outer hypotheses (multi-hypotheses)')
    parser.add_argument('--instances', type=int, default=8, help='M - max. number of instances')
    parser.add_argument('--num_probs', default=4, type=int, help='Q - number of sampling weight maps')
    parser.add_argument('--min_prob', type=float, default=1e-8, help='min sampling weight to avoid degenerate distributions')
    parser.add_argument('--mss', default=6, type=int, help='minimal set size')
    ## GPU usage:
    parser.add_argument('--depth_gpu', default="0",  help='GPU ID for the depth estimator (-1 for CPU)')
    parser.add_argument('--consac_gpu', default="0",  help='GPU ID for the sampling weight estimator (-1 for CPU)')
    parser.add_argument('--inlier_gpu', default="0",  help='GPU ID for inlier computation (-1 for CPU)')
    parser.add_argument('--fitting_gpu', default="0",  help='GPU ID for cuboid parameter fitting (-1 for CPU)')
    ## neural networks:
    parser.add_argument('--load', default="", type=str, help='sampling weight estimator: load pretrained NN weights from file')
    parser.add_argument('--load_depth', default="./models/bts_weights.net", type=str, help='depth estimator: load pretrained NN weights from file')
    parser.add_argument('--load_solver', default=None, type=str, help='load pretrained NN weights for solver from file')
    parser.add_argument('--depth_model', default="gt", help='type of depth source: bts or gt')
    ## cuboid solver:
    parser.add_argument('--minsolver', default="adam", help='which solver to use: adam or transformer')
    parser.add_argument('--fitting_iterations', default=50, type=int, help='cuboid fitting iterations')
    parser.add_argument('--fitting_lr', default=0.01, type=float, help='cuboid fitting iterations')
    parser.add_argument('--a_min', type=float, default=0.001, help='min. cuboid size (shortest side)')
    parser.add_argument('--a_max', type=float, default=2., help='max. cuboid size (longest side)')
    parser.add_argument('--lbfgs', dest='lbfgs', action='store_true', help='use LBFGS instead of Adam for cuboid fitting')
    ## inlier counting:
    parser.add_argument('--no_occlusion_penalty', dest='no_occlusion_penalty', action='store_true', help='disable occlusion penalty')
    parser.add_argument('--oai_crossover', type=float, default=2, help='tau_c/tau - occlusion penalty crosssover (multiple of tau)')
    parser.add_argument('--inlier_mode_selection', default="occlusion_aware", type=str, help='inlier counting for hypothesis selection: occlusion_aware or normal')
    ## misc:
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--jobid', default=-1, type=int, help='SLURM job ID (for logging/debugging)')
    parser.add_argument('--seqransac', dest='seqransac', action='store_true',  help='Use Sequential RANSAC instead of CONSAC for sampling')


    # training only:
    ## general hyperparameters
    parser.add_argument('--epochs', type=int, default=25, help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=0, help='ID of first training epoch')
    parser.add_argument('--batch', '-bs', type=int, default=2, help='B - batch size')
    parser.add_argument('--samplecount', '-ss', type=int, default=2, help='K - sample count')
    parser.add_argument('--consac_lr', type=float, default=1e-5, help='learning rate for sampling weight NN')
    parser.add_argument('--depth_lr', type=float, default=1e-7, help='learning rate for depth estimation NN')
    parser.add_argument('--solver_lr', type=float, default=1e-7, help='learning rate for solver')
    parser.add_argument('--loss_clamp', type=float, default=0.3, help='clamp absolute value of losses')
    parser.add_argument('--softmax_alpha', type=float, default=10., help='alpha - softmax scale factor')
    parser.add_argument('--occlusion_penalty_schedule', type=float, default=1., help='occlusion penalty schedule for training')
    ## regularisation:
    parser.add_argument('--max_prob_loss', type=float, default=0.01, help='kappa_im - inlier masking regularisation')
    parser.add_argument('--minimise_corr', type=float, default=1., help='kappa_corr - correlation min. regularisation')
    parser.add_argument('--maximise_second_entropy', type=float, default=1., help='kappa_entropy - entropy max. regularisation')
    ## flags
    parser.add_argument('--inlier_mode_loss', default="occlusion_aware", type=str, help='inlier counting for loss computation: occlusion_aware or normal')
    parser.add_argument('--finetune', dest='finetune', action='store_true', help='finetune networks from loaded checkpoints')
    parser.add_argument('--train_depth', dest='train_depth', action='store_true', help='train the depth estimator')
    parser.add_argument('--train_consac', dest='train_consac', action='store_true', help='train the sampling weight estimator')
    parser.add_argument('--train_solver', dest='train_solver', action='store_true', help='train the sampling weight estimator')
    parser.add_argument('--load_optimizer', default=None, type=str, help='load optimizer checkpoint for sample weight network')
    parser.add_argument('--resume', default=None, type=str, help='folder to resume training from')
    ## logging:
    parser.add_argument('--ckpt_dir', default='./results/train', help='directory for storing NN weight checkpoints')
    parser.add_argument('--wandb_group', default="", type=str, help='Weights and Biases logging: group name')
    parser.add_argument('--wandb_dir',  default="./tmp", type=str, help='Weights and Biases logging: folder for offline logging')
    parser.add_argument('--wandb_mode', default="disabled", type=str, help='Weights and Biases logging mode: disabled, online or offline')


    # evaluation only:
    parser.add_argument('--inlier_cutoff', default=-1, type=int, help='Theta - min. number of inliers before termination: determined automatically by default')
    parser.add_argument('--runcount', default=1, type=int, help='number of runs for each image')
    parser.add_argument('--split', default="test", help='dataset split to use')
    ## logging:
    parser.add_argument('--eval_results', default="/tmp/results/cuboids_pami", help='path to save results to')
    parser.add_argument('--visualise', dest='visualise', action='store_true', help='save plots with results for each image')
    parser.add_argument('--save_all', dest='save_all', action='store_true', help='save detailed intermediate results for each image')
    parser.add_argument('--dont_save', dest='dont_save', action='store_true', help='do not save intermediate evaluation results')
    ## misc:
    parser.add_argument('--depth_cache', default="/tmp/depth_cache/cuboids_pami", help='destination folder for depth cache')
    parser.add_argument('--read_cache', dest='read_cache', action='store_true', help='read depth from cache')
    parser.add_argument('--write_cache', dest='write_cache', action='store_true', help='write depth to cache')
    parser.add_argument('--write_cache_only', dest='write_cache_only', action='store_true', help='only write depth to cache')
    parser.add_argument('--sampleid', default=-1, type=int, help='start at image index')
    parser.add_argument('--load_eval_results',default=None,type=str, help='load evaluation data from folder')
    ## EM refinement:
    parser.add_argument('--final_em_iter', type=int, default=0, help='number of final EM iterations')
    parser.add_argument('--step_em_iter', type=int, default=0, help='number of intermediate EM iterations')
    parser.add_argument('--em_init_variance', type=float, default=1e-2, help='EM variance sigma')
    parser.add_argument('--em_lr', type=float, default=5e-4, help='EM learning rate')
    parser.add_argument('--em_distance', default="occluded", type=str)


    # debug:
    parser.add_argument('--no_backward', dest='no_backward', action='store_true', help='disable backward pass')
    parser.add_argument('--noval', dest='noval', action='store_true', help='skip validation phase')
    parser.add_argument('--debugging', dest='debugging', action='store_true', help='')

    return parser.parse_args()


def print_options(opt):
    keys = list(vars(opt).keys())
    keys.sort()
    for arg in keys:
        print(arg, ":", getattr(opt, arg))
