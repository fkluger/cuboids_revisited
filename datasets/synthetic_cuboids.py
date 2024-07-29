import torch
from torch.utils.data import Dataset
import numpy as np
from pytorch3d import transforms


def sample_spherical(ndim=3):
    vec = np.random.randn(ndim)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

class SynthCuboids:

    def __init__(self, epoch_size, mss=9, invert=False, size_range=(0.01, 2.), dist_range=(0.5, 10.), xy_range=(-5., 5.)):

        self.invert = invert

        self.size_range = size_range
        self.dist_range = dist_range
        self.xy_range = xy_range

        self.mss = mss

        self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, key):

        axis_angle = 2*np.pi*(np.random.rand(1)-0.5)*sample_spherical()

        t = np.array([np.random.rand() * (self.xy_range[1] - self.xy_range[0]) + self.xy_range[0],
                      np.random.rand() * (self.xy_range[1] - self.xy_range[0]) + self.xy_range[0],
                      np.random.rand() * (self.dist_range[1] - self.dist_range[0]) + self.dist_range[0]], dtype=np.float32)

        R = transforms.axis_angle_to_matrix(torch.from_numpy(axis_angle)).numpy()

        p1 = (R[0, :] - t, R[0, :] + t)
        p2 = (R[1, :] - t, R[1, :] + t)
        p3 = (R[2, :] - t, R[2, :] + t)

        dx = np.linalg.norm(p1[0]) - np.linalg.norm(p1[1])
        dy = np.linalg.norm(p2[0]) - np.linalg.norm(p2[1])
        dz = np.linalg.norm(p3[0]) - np.linalg.norm(p3[1])

        vss = np.sign(np.array([dx, dy, dz]))

        on1 = p1[0] if vss[0] < 0 else p1[1]
        on2 = p2[0] if vss[1] < 0 else p2[1]
        on3 = p3[0] if vss[2] < 0 else p3[1]
        on1 /= np.linalg.norm(on1)
        on2 /= np.linalg.norm(on2)
        on3 /= np.linalg.norm(on3)

        n1 = R[0, :]
        n2 = R[1, :]
        n3 = R[2, :]

        a1 = np.abs(np.dot(on1, n1))
        a2 = np.abs(np.dot(on2, n2))
        a3 = np.abs(np.dot(on3, n3))

        size = np.random.rand(3)*(self.size_range[1]-self.size_range[0])+self.size_range[0]
        side_probs = np.array([size[1]*size[2]*a1, size[0]*size[2]*a2, size[0]*size[1]*a3])
        side_probs /= np.sum(side_probs)

        side_samples = np.random.multinomial(self.mss, side_probs, size=1)[0]

        sides = []
        for si, sn in enumerate(side_samples):
            sides += [si for _ in range(sn)]

        normed_samples = 2.*np.random.rand(self.mss, 2)-1

        samples = np.zeros((self.mss, 3), dtype=np.float32)

        for i in range(self.mss):
            if sides[i] == 0:
                samples[i] = [vss[0]*size[0], normed_samples[i, 0]*size[1], normed_samples[i, 1]*size[2]]
            elif sides[i] == 1:
                samples[i] = [normed_samples[i, 0]*size[0], vss[1]*size[1], normed_samples[i, 1]*size[2]]
            elif sides[i] == 2:
                samples[i] = [normed_samples[i, 0]*size[0], normed_samples[i, 1]*size[1], vss[2]*size[2]]

        samples = samples @ R + t[None, :]

        samples = samples.astype(np.float32)

        return size, axis_angle, t, samples



class SynthCuboidsDataset(Dataset):

    def __init__(self, epoch_size, **kwargs):
        self.dataset = SynthCuboids(epoch_size, **kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        size, axis_angle, t, samples = self.dataset[key]

        return size, axis_angle, t, samples

# import open3d as o3d


def cuboid_mesh(ax, ay, az, R, t, colour=None):
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh.create_box(width=2, height=2, depth=2)
    vertices = np.asarray(mesh.vertices) - 1.
    vertices[:, 0] *= ax
    vertices[:, 1] *= ay
    vertices[:, 2] *= az
    vertices = vertices @ R
    vertices[:, 0] += t[0]
    vertices[:, 1] += t[1]
    vertices[:, 2] += t[2]

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()

    c = np.array([0, 0, 1], dtype=np.float64) if colour is None else colour.astype(np.float64)
    mesh.paint_uniform_color(c)

    return mesh


if __name__ == "__main__":
    import open3d as o3d

    dataset = SynthCuboids(100)

    for key in range(len(dataset)):
        size, axis_angle, t, samples = dataset[key]

        R = transforms.axis_angle_to_matrix(torch.from_numpy(axis_angle)).numpy()

        cuboid = cuboid_mesh(size[0], size[1], size[2], R, t)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(samples)
        pcd.paint_uniform_color([0, 1, 0])

        origin = o3d.geometry.PointCloud()
        origin.points = o3d.utility.Vector3dVector(np.array([[0., 0, 0]]))
        origin.paint_uniform_color([1, 0, 0])

        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(np.concatenate([np.array([[0,0,0]]), samples], axis=0))
        line_corr = [[0,i] for i in range(1, samples.shape[0]+1)]
        line_corr = np.array(line_corr)
        lines.lines = o3d.utility.Vector2iVector(line_corr)
        lines.paint_uniform_color([0, 1, 0])

        o3d.visualization.draw_geometries([cuboid, pcd, origin, lines], width=640, height=480)




