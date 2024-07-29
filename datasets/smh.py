import os.path
from torch.utils.data import Dataset
import numpy as np
import pickle
import skimage.io
import skimage.transform
import random
import cv2

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

class SMH:

    def __init__(self, data_dir, split, keep_in_mem=False, normalize_coords=True, return_images=False, shuffle=False, scale=1.0):

        self.data_dir = data_dir
        self.keep_in_mem = keep_in_mem
        self.normalize_coords = normalize_coords
        self.return_images = return_images
        self.scale = scale

        self.img_size = (int(1024*scale), int(1024*scale))

        # self.train_sequences = [1, 2, 3, 4]
        self.train_sequences = [0, 1, 2, 3, 4]
        self.val_sequences = [5]
        self.test_sequences = [6]

        self.pairs = []

        self.split = split
        self.num_samples = None

        if split == "train":
            self.coarse_paths = self.train_sequences
        elif split == "train_random":
            self.coarse_paths = self.train_sequences
            self.num_samples = 500
        elif split == "train_small":
            self.coarse_paths = [self.train_sequences[0]]
        elif split == "val":
            self.coarse_paths = self.val_sequences
        elif split == "val_random":
            self.coarse_paths = self.val_sequences
            self.num_samples = 20
        elif split == "test":
            self.coarse_paths = self.test_sequences
        elif split == "all":
            self.coarse_paths = self.train_sequences + self.val_sequences + self.test_sequences
        else:
            assert False, "invalid split: %s" % split

        os.makedirs("./tmp/smh_pairs", exist_ok=True)
        pairs_cache_file = os.path.join("./tmp/smh_pairs", split+".pkl")
        if os.path.exists(pairs_cache_file):
            with open(pairs_cache_file, 'rb') as f:
                self.pairs = pickle.load(f)
        else:
            for coarse_path in self.coarse_paths:
                for fine_path_dir in os.scandir(os.path.join(self.data_dir, "%d" % coarse_path)):
                    if fine_path_dir.is_dir():
                        print(fine_path_dir.path)
                        for pair_path_dir in os.scandir(fine_path_dir.path):
                            if pair_path_dir.is_dir():
                                if os.path.exists(os.path.join(pair_path_dir.path, "features_and_ground_truth.npz")):
                                    features_and_gt = np.load(os.path.join(pair_path_dir.path, "features_and_ground_truth.npz"),
                                                              allow_pickle=True)


                                    gt_label = features_and_gt["labels"]
                                    num_homographies = np.max(gt_label)
                                    if num_homographies > 2:

                                        split_path = pair_path_dir.path.split("/")
                                        coarse = split_path[-3]
                                        fine = split_path[-2]
                                        pair = split_path[-1]

                                        # self.pairs += [pair_path_dir.path]
                                        self.pairs += [(coarse, fine, pair)]
            self.pairs.sort()
            with open(pairs_cache_file, 'wb') as f:
                pickle.dump(self.pairs, f, pickle.HIGHEST_PROTOCOL)
        # print(self.pairs)

        if shuffle:
            random.shuffle(self.pairs)

        print("%s dataset: %d samples" % (split, len(self.pairs)))

        self.cache_dir = None
        if keep_in_mem:
            cache_folders = ["/phys/ssd/tmp/city_cuboids", "/phys/ssd/slurmstorage/tmp/city_cuboids",
                             "/tmp/city_cuboids", "/phys/intern/tmp/city_cuboids"]
            for cache_folder in cache_folders:
                try:
                    cache_folder = os.path.join(cache_folder, split)
                    os.makedirs(cache_folder, exist_ok=True)
                    self.cache_dir = cache_folder
                    print("%s is cache folder" % cache_folder)
                    break
                except:
                    print("%s unavailable" % cache_folder)

    def __len__(self):
        if self.num_samples is not None:
            return self.num_samples
        return len(self.pairs)

    def __getitem__(self, key):

        # folder = self.pairs[key]
        if "random" in self.split:
            if key >= self.__len__():
                raise IndexError
            coarse, fine, pair = random.choice(self.pairs)
        else:
            coarse, fine, pair = self.pairs[key]
        folder = os.path.join(self.data_dir, coarse, fine, pair)

        datum = None

        if self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, "%09d.pkl" % key)
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    datum = pickle.load(f)
                depth_path = os.path.join(self.cache_dir, "%09d.png" % key)
                # depth_read = imageio.imread(depth_path)
                depth_read = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if datum is None:
            # folder2 = folder.replace("processed2", "processed_depth")
            features_and_gt = np.load(os.path.join(folder, "features_and_ground_truth.npz"), allow_pickle=True)

            depth_path = os.path.join(folder, "depth1.png")
            # depth_read = imageio.imread(depth_path)
            depth_read = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            # depth = features_and_gt["depth1"]
            intrinsic = features_and_gt["K1"]

            img1_path = os.path.join(folder, "render1.png")
            if os.path.exists(img1_path):
                image1_rgb = skimage.io.imread(img1_path).astype(float)[:, :, :3]
            else:
                image1_rgb = None

            datum = {'img1': image1_rgb, 'intrinsic':intrinsic,
                     'img1size': self.img_size, 'img2size': self.img_size}

            if self.cache_dir is not None:
                cache_path = os.path.join(self.cache_dir, "%09d.pkl" % key)
                if not os.path.exists(cache_path):
                    with open(cache_path, 'wb') as f:
                        pickle.dump(datum, f, pickle.HIGHEST_PROTOCOL)

                depth_path = os.path.join(self.cache_dir, "%09d.png" % key)
                # imageio.imwrite(depth_path, depth_read)
                cv2.imwrite(depth_path, depth_read)

        datum["depth_read"] = depth_read

        mask = (datum["depth_read"] < 65535)
        datum["mask"] = np.array(mask).astype(np.float32)

        depth = datum["depth_read"].astype(np.float32) / (65535.0 / 1000.0)

        intrinsic = datum["intrinsic"]
        intrinsic[1,1] *= -1

        if self.scale < 1:
            depth_ = skimage.transform.resize(depth, (int(depth.shape[0] * self.scale),
                                                      int(depth.shape[1] * self.scale)), anti_aliasing=False, order=0)

            datum["mask"] = skimage.transform.resize(datum["mask"], (int(depth.shape[0] * self.scale),
                                                      int(depth.shape[1] * self.scale)), anti_aliasing=False, order=0)

            scale = np.array([[self.scale, 0, 0],
                              [0, self.scale, 0],
                              [0, 0, 1]])
            intrinsic = scale @ intrinsic
            if datum["img1"] is not None:
                datum["img1"] = skimage.transform.resize(datum["img1"], self.img_size)
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

        datum["depth"] = depth_
        datum["coord_grid"] = coord_grid_.astype(np.float32)

        datum["intrinsic"] = intrinsic

        return datum


class SMHDataset(Dataset):

    def __init__(self, data_dir_path, split, keep_in_mem=True, scale=1.0):
        self.homdata = SMH(data_dir_path, split, keep_in_mem=keep_in_mem, scale=scale)
        self.split = split
        self.scale = scale

        self.depth_mean = 35.42254
        self.depth_stdd = np.sqrt(1143.3948)
        self.image_mean = np.array([110.53406,  110.92176,  109.770515], dtype=np.float32) / 255.0
        self.image_stdd = np.sqrt(np.array([2426.1313, 2379.8718, 2347.3743], dtype=np.float32)) / 255.0


    def denormalise(self, X):
        scale = np.max(self.homdata.img_size) / 2.0
        off = (self.homdata.img_size[1] / 2.0, self.homdata.img_size[0] / 2.0)
        p1 = X[..., 0:2] * scale
        p2 = X[..., 0:2] * scale
        p1[..., 0] += off[0]
        p1[..., 1] += off[1]
        p2[..., 0] += off[0]
        p2[..., 1] += off[1]

        return p1, p2

    def get_image_size(self):
        return (int(1024*self.scale), int(1024*self.scale))

    def __len__(self):
        return len(self.homdata)

    def __getitem__(self, key):
        datum = self.homdata[key]

        if 'img1' in datum.keys() and datum['img1'] is not None:
            image = datum['img1'].astype(np.float32) / 255.0
        else:
            image = 0

        depth = datum["depth"].astype(np.float32)
        intrinsic = datum["intrinsic"].astype(np.float32)
        coord_grid = datum["coord_grid"].astype(np.float32)
        labels = np.zeros(depth.shape[0:2])
        model_params = np.zeros((1, 12))

        return image, intrinsic, coord_grid, labels, model_params, depth, 0, datum["mask"]

def get_statistics():
    # trainset = SMHDataset("/data/kluger/datasets/city4/processed_depth", split='all', scale=1.0)
    # exit(0)
    trainset = SMHDataset("/data/kluger/datasets/city4/processed2", split='train', scale=1.0)

    image_means = []
    depth_means = []
    depth_min = None
    depth_max = None
    for image, intrinsic, coord_grid, labels, model_params, depth, _ in trainset:
        image = image.astype(float) / 255.0
        image_means += [np.mean(image, axis=(0,1))]
        depth_means += [np.mean(depth[np.nonzero(depth < 999)])]
        if depth_min is None:
            depth_min = np.min(depth)
        else:
            depth_min = np.minimum(np.min(depth), depth_min)
        if depth_max is None:
            depth_max = np.max(depth[np.nonzero(depth < 999)])
        else:
            depth_max = np.maximum(np.max(depth[np.nonzero(depth < 999)]), depth_max)

    image_mean = np.mean(np.stack(image_means, axis=-1), axis=-1)
    print("image mean: ", image_mean)
    depth_mean = np.mean(depth_means)
    print("depth mean: ", depth_mean)
    print("depth min: ", depth_min)
    print("depth max: ", depth_max)

    image_means = []
    depth_means = []
    for image, intrinsic, coord_grid, labels, model_params, depth, _ in trainset:
        image = image.astype(float) / 255.0
        image_means += [np.mean((image-image_mean[None, None, :])**2, axis=(0,1))]
        depth_means += [np.mean((depth[np.nonzero(depth < 999)]-depth_mean)**2)]

    image_var = np.mean(np.stack(image_means, axis=-1), axis=-1)
    print("image var: ", image_var)
    depth_var = np.mean(depth_means)
    print("depth var: ", depth_var)


def make_vis():
    import matplotlib
    import matplotlib.pyplot as plt

    random.seed(0)

    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    dataset = SMH("../datasets/smh", "all", keep_in_mem=False, normalize_coords=False, return_images=True, shuffle=True)

    target_folder = "./tmp/fig/smh"

    os.makedirs(target_folder, exist_ok=True)

    for idx in range(len(dataset)):
        print("%d / %d" % (idx+1, len(dataset)), end="\r")
        sample = dataset[idx]
        img1 = sample["img1"].astype(np.uint8)
        img2 = sample["img2"].astype(np.uint8)
        pts1 = sample["points1"]
        pts2 = sample["points2"]
        y = sample["labels"]

        num_models = np.max(y)
        N = pts1.shape[0]

        hues = np.array([0, 210, 45, 170, 100, 310, 255, 80, 190, 280, 120, 340, 295, 60, 220, 25, 150, 320, 230, 190, 290, 130, 350, 270, 30, 140, 200, 300])

        cb_hex = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#8e10b3", "#374009", "#aec8ea", "#56611b", "#64a8c6", "#99d8d4", "#745a50", "#46fa50", "#e09eea", "#5b2b1f", "#723f91", "#634418", "#7db0d0", "#1ae37c", "#aa462c", "#719bb7", "#463aa2", "#98f42e", "#32185b", "#364fcd", "#7e54c8", "#bb5f7f", "#d466d5", "#5a0382", "#443067", "#a76232", "#78dbc1", "#35a4b2", "#52d387", "#af5a7e", "#3ef57d", "#d6d993"]
        cb = np.array([matplotlib.colors.to_rgb(x) for x in cb_hex])

        fig = plt.figure(figsize=(4 * 4, 4 * 2), dpi=256)
        axs = fig.subplots(nrows=1, ncols=2)
        for ax in axs:
            ax.set_aspect('equal', 'box')
            ax.axis('off')

        img1g = rgb2gray(img1) * 0.5 + 128
        img2g = rgb2gray(img2) * 0.5 + 128

        axs[0].imshow(img1g, cmap='Greys_r', vmin=0, vmax=255)
        axs[1].imshow(img2g, cmap='Greys_r', vmin=0, vmax=255)

        for j, pts in enumerate([pts1, pts2]):
            ax = axs[j]

            h = hues[y]/360.0
            s = np.ones_like(h)
            v = (y > 0).astype(float)
            c = matplotlib.colors.hsv_to_rgb(np.stack([h, s, v], axis=1))

            ms = np.where(y>0, 8, 4)

            c = cb[y]

            ax.scatter(pts[:, 0], pts[:, 1], c=c, s=ms**2)

        fig.tight_layout()
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(os.path.join(target_folder, "%02d_%03d_vis.png" % (num_models, idx)), bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        fig = plt.figure(figsize=(4 * 4, 4 * 2), dpi=150)
        axs = fig.subplots(nrows=1, ncols=2)
        for ax in axs:
            ax.set_aspect('equal', 'box')
            ax.axis('off')

        axs[0].imshow(img1)
        axs[1].imshow(img2)
        fig.tight_layout()
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(os.path.join(target_folder, "%02d_%03d_orig.png" % (num_models, idx)), bbox_inches='tight',
                    pad_inches=0)
        plt.close()



if __name__ == "__main__":
    # make_vis()
    get_statistics()
