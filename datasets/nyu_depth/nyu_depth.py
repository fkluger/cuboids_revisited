import os.path
import scipy.io
from torch.utils.data import Dataset
import numpy as np
import pickle
import skimage.transform
import time


class NYUDepth:

    def __init__(self, keep_in_mem=True, data_directory=None, split='train', scale=1., split_mat=None):

        self.keep_in_mem = keep_in_mem

        self.dataset_files = None

        self.scale = scale

        self.image_size = (640, 480)

        if data_directory is not None:
            # files = glob.glob(data_directory + "/*.pkl")
            files = [os.path.join(data_directory, "%04d.pkl" % i) for i in range(1449)]
            files.sort()

            print("read folder contents")

            if split_mat is None:
                if split == 'train':
                    # self.dataset_files = files[:1000]
                    self.dataset_files = [(i, files[i]) for i in range(1000)]
                elif split == 'val':
                    # self.dataset_files = files[1000:1224]
                    self.dataset_files = [(i, files[i]) for i in range(1000, 1224)]
                elif split == 'test':
                    # self.dataset_files = files[1224:]
                    self.dataset_files = [(i, files[i]) for i in range(1224, len(files))]
                elif split == 'all':
                    # self.dataset_files = files
                    self.dataset_files = [(i, files[i]) for i in range(len(files))]
                else:
                    assert False
            else:
                train_test = scipy.io.loadmat(split_mat)
                test_images = [int(x)-1 for x in train_test["testNdxs"]]
                train_images = [int(x)-1 for x in train_test["trainNdxs"]]

                if split == 'train':
                    self.dataset_files = [(i, files[i]) for i in train_images[0:600]]
                elif split == 'val':
                    self.dataset_files = [(i, files[i]) for i in train_images[600:]]
                elif split == 'trainval':
                    self.dataset_files = [(i, files[i]) for i in train_images]
                elif split == 'test':
                    self.dataset_files = [(i, files[i]) for i in test_images]
                elif split == 'all':
                    self.dataset_files = [(i, files[i]) for i in test_images+train_images]
                else:
                    assert False

            self.dataset = [None for _ in self.dataset_files]
        else:
            assert False

        fx_rgb = 5.1885790117450188e+02
        fy_rgb = 5.1946961112127485e+02
        cx_rgb = 3.2558244941119034e+02
        cy_rgb = 2.5373616633400465e+02

        self.K = np.array([[fx_rgb, 0, cx_rgb], [0, fy_rgb, cy_rgb], [0, 0, 1]])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):

        datum = self.dataset[key]

        if datum is None:

            try_loading = True
            wait_time = 1
            tries = 0

            file_idx, filename = self.dataset_files[key]

            while try_loading:
                if os.path.isfile(filename):
                    with open(filename, 'rb') as f:
                        data = pickle.load(f)
                    try_loading = False
                else:
                    if tries < 10:
                        print("could not open file, wait %d seconds, retry.." % wait_time)
                        time.sleep(wait_time)
                        wait_time *= 2
                        tries += 1
                    else:
                        assert False, "could not open file %s" % filename

            depth = data["depth"]
            image = data["image"]
            intrinsic = self.K

            if self.scale < 1:
                depth_ = skimage.transform.resize(depth, (int(depth.shape[0] * self.scale),
                                                         int(depth.shape[1] * self.scale)), anti_aliasing=True)

                scale = np.array([[self.scale, 0, 0],
                                  [0, self.scale, 0],
                                  [0, 0, 1]])
                intrinsic = scale @ self.K
            else:
                depth_ = depth

            width = depth_.shape[1]
            height = depth_.shape[0]
            coord_grid = np.ones((height, width, 3), dtype=np.float32)

            x = np.arange(0, width)
            y = np.arange(0, height)
            xv, yv = np.meshgrid(x, y)
            coord_grid[:, :, 0] = xv
            coord_grid[:, :, 1] = yv

            Kinv = np.linalg.inv(intrinsic)
            coord_grid_ = np.transpose(Kinv @ np.transpose(coord_grid, (0, 2, 1)), (0, 2, 1))
            coord_grid_[:, :, 0] *= depth_ / coord_grid_[:, :, 2]
            coord_grid_[:, :, 1] *= depth_ / coord_grid_[:, :, 2]
            coord_grid_[:, :, 2] *= depth_ / coord_grid_[:, :, 2]

            datum = {'depth': depth, 'intrinsic': intrinsic, "coord_grid": coord_grid_.astype(np.float32),
                     'image': image, 'index': file_idx}

            if self.keep_in_mem:
                self.dataset[key] = datum

        return datum


class NYUDepthDataset(Dataset):

    def __init__(self, max_num_sq=3, **kwargs):
        self.dataset = NYUDepth(**kwargs)
        self.max_num_sq = max_num_sq

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        datum = self.dataset[key]

        intrinsic = self.dataset.K.astype(np.float32)
        depth = datum["depth"].astype(np.float32)
        coord_grid = datum["coord_grid"]
        labels = np.zeros(depth.shape[0:2])
        model_params = np.zeros((1, 12))

        return depth, intrinsic, coord_grid, labels, model_params

    def get_image_size(self, key=0):
        datum = self.dataset[key]
        depth = datum["depth"]
        return depth.shape[0:2]


class NYURGBDataset(Dataset):

    def __init__(self, max_num_sq=3, **kwargs):
        self.dataset = NYUDepth(**kwargs)
        self.max_num_sq = max_num_sq

        self.image_mean = np.array([0.48936939, 0.42036757, 0.39901595], dtype=np.float32)
        self.depth_mean = 2.7163546
        self.image_stdd = np.sqrt(np.array([0.2884224, 0.29518136, 0.3076983], dtype=np.float32))
        self.depth_stdd = np.sqrt(1.3985424)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        datum = self.dataset[key]

        intrinsic = self.dataset.K.astype(np.float32)
        depth = datum["depth"].astype(np.float32)
        image = datum["image"].astype(np.float32) / 255.
        coord_grid = datum["coord_grid"]
        labels = np.zeros(depth.shape[0:2])
        model_params = np.zeros((1, 12))

        mask = np.ones(coord_grid.shape[:2], dtype=np.float32)

        return image, intrinsic, coord_grid, labels, model_params, depth, datum["index"], mask

    def get_image_size(self, key=0):
        datum = self.dataset[key]
        depth = datum["depth"]
        return depth.shape[0:2]


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from matplotlib import ticker
    plt.rcParams['text.usetex'] = True

    target_folder = "/home/kluger/tnt/thesis/chap_4_datasets/fig/nyu_depth"
    os.makedirs(target_folder, exist_ok=True)

    dataset = NYUDepth(keep_in_mem = True, data_directory = "/data/kluger/datasets/nyu_files", split = 'all', scale = 1., split_mat = None)

    for idx, sample in enumerate(dataset):
        print(idx)
        fig = plt.figure(figsize=(4*2+0.5, 3), dpi=256)
        axs = fig.subplots(nrows=1, ncols=3, gridspec_kw={'width_ratios': [8, 8, 1]})
        for ax in axs[:2]:
            ax.set_aspect('equal', 'box')
            ax.axis('off')

        axs[0].imshow(sample["image"])
        pos = axs[1].imshow(sample["depth"], cmap="cividis")

        bar = fig.colorbar(pos, cax=axs[2])
        # bar.set_label('depth (m)', rotation=270)
        tick_locator = ticker.MaxNLocator(nbins=10, min_n_ticks=8)
        bar.locator = tick_locator
        bar.update_ticks()

        fig.tight_layout()
        axs[0].set_axis_off()
        axs[1].set_axis_off()
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.savefig(os.path.join(target_folder, "%04d_vis.png" % (idx)), bbox_inches='tight', pad_inches=0)
        plt.savefig(os.path.join(target_folder, "%04d_vis.pdf" % (idx)), pad_inches=0)
        plt.close()
        # plt.show()

        # exit(0)
