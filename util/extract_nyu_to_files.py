import scipy.io
import pickle
import os
import argparse

parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--source', default="./nyu_depth_v2_labeled.matv7.mat", help='path to .mat file')
parser.add_argument('--destination', default="./datasets/nyu_depth/files", help='path to destination folder')

opt = parser.parse_args()

mat_file = opt.source
target_folder = opt.destination

data_mat = scipy.io.loadmat(mat_file)

all_depths = data_mat['depths']
all_images = data_mat['images']

num_images = all_images.shape[-1]

for i in range(num_images):
    depth = all_depths[:, :, i]
    image = all_images[:, :, :, i]

    pickle_file = os.path.join(target_folder, "%04d.pkl" % i)

    data = {'image': image, 'depth': depth}

    pickle.dump(data, open(pickle_file, "wb"))

    print(i)

