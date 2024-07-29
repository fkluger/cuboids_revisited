import torch
import numpy as np
from pyquaternion import Quaternion
import pytorch3d as p3d
import pytorch3d.transforms
import math
import sklearn.metrics
import scipy.spatial


def sigmoid(x):
    return 1. / (1. + torch.exp(-x))

def sigmoid_(x):
    return 1. / (1. + math.exp(-x))


def soft_inlier_fun_gen(beta, tau):
    def f(d):
        return 1 - sigmoid(beta * d - beta * tau)
    return f


def soft_outlier_fun_gen(beta, tau):
    def f(d):
        return sigmoid(beta * d - beta * tau)
    return f

def soft_outlier_lin_fun_gen(beta, tau, crossover):
    sig_x = beta*crossover - beta*tau
    gradient = beta * sigmoid_(sig_x) * (1-sigmoid_(sig_x))
    value = sigmoid_(sig_x)

    def f(d):
        sig_part = sigmoid(beta * d - beta * tau)
        lin_part = gradient * d + (value-gradient*crossover)
        return torch.where(d < crossover, sig_part, lin_part)
        # return sigmoid(beta * d - beta * tau)
    return f


def soft_inlier_fun(d, beta, tau):
    return 1 - sigmoid(beta * d - beta * tau)


def auc_loss(values, threshold):
    P, S, K, B, Y = values.size()

    values_batched = values.view(-1, Y)
    sorted_values, _ = torch.sort(values_batched)
    inlier_range = torch.range(1./Y, 1, 1./Y).to(values.device).expand(sorted_values.size(0), -1)

    valid_values = sorted_values * (sorted_values <= threshold).float()
    max_valid_values, max_valid_indices = torch.max(valid_values[:, :inlier_range.size(-1)], -1)

    max_valid_inlier = torch.gather(inlier_range, 1, max_valid_indices.unsqueeze(-1)).squeeze(-1)
    max_valid_area = (threshold-max_valid_values) * max_valid_inlier

    value_diffs = valid_values[:, 1:] - valid_values[:, :-1]

    areas = value_diffs * inlier_range

    max_invalid_area = torch.gather(areas, 1, max_valid_indices.unsqueeze(-1)).squeeze()

    area_sum = torch.sum(areas, -1) - max_invalid_area + max_valid_area

    auc = area_sum / threshold

    auc = auc.view(P, S, K, B)

    return auc


def cuboid_distance_batched(params, points, reduce=False, weighted=False, a_max=100., a_min=0.01, norm_by_volume=False, verbose=False, sigmoid_size=False):

    params = torch.where(params != params, torch.zeros_like(params), params)

    if sigmoid_size:
        a1 = torch.sigmoid(params[:, 0]).unsqueeze(-1) * (a_max-a_min) + a_min
        a2 = torch.sigmoid(params[:, 1]).unsqueeze(-1) * (a_max-a_min) + a_min
        a3 = torch.sigmoid(params[:, 2]).unsqueeze(-1) * (a_max-a_min) + a_min
    else:
        a1 = torch.clamp(params[:, 0], a_min, a_max).unsqueeze(-1)
        a2 = torch.clamp(params[:, 1], a_min, a_max).unsqueeze(-1)
        a3 = torch.clamp(params[:, 2], a_min, a_max).unsqueeze(-1)

    angle_axis = torch.clamp(params[:, 3:6], -100, 100)
    t = torch.clamp(params[:, 6:9], -100, 100)

    if torch.isnan(angle_axis).any():
        assert False

    R = p3d.transforms.axis_angle_to_matrix(angle_axis)

    t_ = t.unsqueeze(1)
    points_ = points[:, :, :3]
    points_transformed = points_ - t_
    points_transformed = torch.matmul(points_transformed, R.transpose(-2, -1))
    # points_transformed = torch.matmul(points_transformed, R) #here

    x = points_transformed[:, :, 0]
    y = points_transformed[:, :, 1]
    z = points_transformed[:, :, 2]

    zero = torch.FloatTensor([0]).to(x.device)

    outside_values = torch.max(torch.abs(x)-a1, zero)**2 + \
                     torch.max(torch.abs(y)-a2, zero)**2 + \
                     torch.max(torch.abs(z)-a3, zero)**2
    inside_values = torch.max(torch.min(a1-torch.abs(x),
                              torch.min(a2-torch.abs(y),
                                        a3-torch.abs(z))), zero)**2

    squared_distances = outside_values-inside_values
    squared_distances = torch.abs(squared_distances)

    if weighted:
        weights = points[:, :, 3]
        squared_distances = squared_distances * weights

    if norm_by_volume:
        # squared_distances = squared_distances + 0.01 * (a1 + a2 + a3)
        squared_distances = squared_distances * (a1 + a2 + a3)

    if reduce:
        return torch.sum(squared_distances, dim=-1)
    else:
        return squared_distances


def cuboid_angle_constraint(params):

    params = torch.where(params != params, torch.zeros_like(params), params)

    B = params.size(0)
    q = torch.clamp(params[:, 3:6], -100, 100)

    if torch.isnan(q).any():
        assert False

    angle_axis = q
    R = p3d.transforms.axis_angle_to_matrix(angle_axis)

    R1 = R.unsqueeze(0)
    R2 = R.unsqueeze(1).transpose(-1, -2)

    RRT = torch.matmul(R1, R2).view(B, B, -1)

    RRT_ = torch.matmul(R1, R2).view(-1, 3, 3)

    residual_angle_axis = p3d.transforms.quaternion_to_axis_angle(p3d.transforms.matrix_to_quaternion(RRT_))
    residual_angles = torch.norm(residual_angle_axis, dim=-1) * 180. / torch.pi
    residual_angles_ = residual_angles % 90

    # print("angle loss: " , residual_angles_.cpu().detach().numpy())

    return residual_angles_.mean()
    # max_values, _ = torch.max(RRT**2, -1)

    # return torch.mean(1-max_values)


def cuboid_fun_numpy(params, points, norm_by_volume, reduce=True, a_max=2., a_min=0.01):

    a1 = np.clip(params[0], a_min, a_max)
    a2 = np.clip(params[1], a_min, a_max)
    a3 = np.clip(params[2], a_min, a_max)

    q = np.clip(params[3:6], -100, 100)
    t = np.clip(params[6:9], -100, 100)

    points_ = points.copy()
    points_[:, 0] -= t[0]
    points_[:, 1] -= t[1]
    points_[:, 2] -= t[2]

    angle = np.linalg.norm(q)
    angle_axis = Quaternion(axis=q/angle, angle=angle)
    R = angle_axis.rotation_matrix

    points_transformed = points_ @ R.T

    x = points_transformed[:, 0]
    y = points_transformed[:, 1]
    z = points_transformed[:, 2]

    outside_values = np.maximum(np.abs(x)-a1, 0)**2 + \
                     np.maximum(np.abs(y)-a2, 0)**2 + \
                     np.maximum(np.abs(z)-a3, 0)**2
    inside_values = np.maximum(np.minimum(a1-np.abs(x),
                              np.minimum(a2-np.abs(y),
                                        a3-np.abs(z))), 0)**2

    squared_distances = outside_values-inside_values

    squared_distances = np.abs(squared_distances)

    if norm_by_volume:
        squared_distances = squared_distances * (a1 + a2 + a3)

    if reduce:
        return np.sum(squared_distances, axis=-1)
    else:
        return squared_distances


def cuboid_occlusion_batched(params, points, a_max=100., a_min=0.001):

    # params = torch.where(params != params, torch.zeros_like(params), params)

    B = params.size(0)
    Y = points.size(1)

    a1 = torch.clamp(params[:, 0], a_min, a_max)
    a2 = torch.clamp(params[:, 1], a_min, a_max)
    a3 = torch.clamp(params[:, 2], a_min, a_max)
    angle_axis = params[:, 3:6]
    t = params[:, 6:9]

    R = pytorch3d.transforms.axis_angle_to_matrix(angle_axis)

    points_transformed = points[:, :, :3] - t[:, None, :]
    points_transformed = torch.matmul(points_transformed, R.transpose(-2, -1))# BxYx3
    # points_transformed = torch.matmul(points_transformed, R)#here
    x = points_transformed[:, :, 0]
    y = points_transformed[:, :, 1]
    z = points_transformed[:, :, 2]

    origins_transformed_ = -t
    origins_transformed = torch.matmul(origins_transformed_.unsqueeze(1), R.transpose(-2, -1)) # Bx1x3
    # origins_transformed = torch.matmul(origins_transformed_.unsqueeze(1), R) #here

    support = points_transformed
    vectors = origins_transformed-points_transformed
    vector_norms = torch.norm(vectors, dim=-1, keepdim=True)
    # vectors /= vector_norms -> this results in a GPU memory leak, why?
    vectors = vectors / vector_norms

    plane_normals = torch.zeros((1, 6, 3)).to(params.device)
    plane_normals[:, 0:2, 0] = 1
    plane_normals[:, 2:4, 1] = 1
    plane_normals[:, 4:6, 2] = 1

    plane_dists = torch.stack([a1, -a1, a2, -a2, a3, -a3], dim=-1)

    plane_dists_ = plane_dists.unsqueeze(1) # Bx1x6

    lambdas = (plane_dists_ - torch.matmul(support, plane_normals.transpose(-1, -2))) \
              / torch.matmul(vectors, plane_normals.transpose(-1, -2)) # BxYx6

    lambdas_ = lambdas.unsqueeze(-1) # BxYx6x1
    support_ = support.unsqueeze(2)  # BxYx1x3
    vectors_ = vectors.unsqueeze(2)  # BxYx1x3

    intersections = support_ + lambdas_ * vectors_  # BxYx6x3

    a1_ = a1.unsqueeze(-1)
    a2_ = a2.unsqueeze(-1)
    a3_ = a3.unsqueeze(-1)  # Bx1

    distances_to_a1_pos = (x-a1_)**2 + torch.clamp(torch.abs(y)-a2_, min=0)**2 + torch.clamp(torch.abs(z)-a3_, min=0)**2
    distances_to_a1_neg = (-x-a1_)**2 + torch.clamp(torch.abs(y)-a2_, min=0)**2 + torch.clamp(torch.abs(z)-a3_, min=0)**2
    distances_to_a2_pos = (y-a2_)**2 + torch.clamp(torch.abs(x)-a1_, min=0)**2 + torch.clamp(torch.abs(z)-a3_, min=0)**2
    distances_to_a2_neg = (-y-a2_)**2 + torch.clamp(torch.abs(x)-a1_, min=0)**2 + torch.clamp(torch.abs(z)-a3_, min=0)**2
    distances_to_a3_pos = (z-a3_)**2 + torch.clamp(torch.abs(x)-a1_, min=0)**2 + torch.clamp(torch.abs(y)-a2_, min=0)**2
    distances_to_a3_neg = (-z-a3_)**2 + torch.clamp(torch.abs(x)-a1_, min=0)**2 + torch.clamp(torch.abs(y)-a2_, min=0)**2

    distances_to_cube_sides = torch.stack([distances_to_a1_pos, distances_to_a1_neg,
                                           distances_to_a2_pos, distances_to_a2_neg,
                                           distances_to_a3_pos, distances_to_a3_neg], dim=-1) # BxYx6

    dist_values = params[:, 0:3][:, None, None, :]-torch.abs(intersections)

    intersections_on_cube = (dist_values >= -1e-2).float()

    intersections_on_cube_ = intersections_on_cube[:, :, :, 0] *intersections_on_cube[:, :, :, 1]*intersections_on_cube[:, :, :, 2]

    possible_occlusion = (lambdas > 0).float()

    occlusions = possible_occlusion * intersections_on_cube_

    return occlusions, distances_to_cube_sides


def cuboid_distance_occluded(models, data, weighted=False, reduce=False, norm_by_volume=False, **kwargs):

    occlusions, distances_to_cube_sides = \
        cuboid_occlusion_batched(models, data, **kwargs)
    occluded_distances = torch.max(occlusions * distances_to_cube_sides, dim=-1)[0]
    min_distances = torch.min(distances_to_cube_sides, dim=-1)[0]
    oa_distances = torch.max(occluded_distances, min_distances)

    if weighted:
        weights = data[:, :, 3]
        oa_distances *= weights

    if norm_by_volume:
        a1 = models[:, 0].unsqueeze(1)
        a2 = models[:, 1].unsqueeze(1)
        a3 = models[:, 2].unsqueeze(1)
        oa_distances = oa_distances * (a1 + a2 + a3)

    if reduce:
        return torch.sum(oa_distances, dim=-1)
    else:
        return oa_distances


def gric(errors_sq, sigma_sq, d, k, D, prior, ndiv):
    n = torch.numel(errors_sq)
    lambda1 = math.log(n / (ndiv*2*math.pi*sigma_sq))
    lambda2 = math.log(n)
    T = 2*math.log(prior/(1-prior)) + (D-d)*lambda1

    rho = errors_sq/sigma_sq
    rho = torch.clamp(rho, 0, T)


    return torch.sum(rho) + lambda1 * d * n + lambda2 * k


def calc_auc_values_np(distances_np):
    max_auc_distances = [50, 20, 10, 5]
    Y = distances_np.size
    mean_distance = np.mean(distances_np)
    all_distances_sorted = np.sort(distances_np)
    inlier_range = np.arange(Y).astype(np.float32) * 1. / Y

    sample_auc_values = {}
    for max_distance in max_auc_distances:
        max_value = max_distance / 100.
        x = np.append(all_distances_sorted[np.where(all_distances_sorted < max_value)], max_value)
        y = inlier_range[np.where(all_distances_sorted < max_value)]
        if y.size > 2:
            y = np.append(y, y[-1])

            auc = sklearn.metrics.auc(x, y) / max_value
        else:
            auc = 0

        sample_auc_values["auc_at_%d" % max_distance] = auc

    sample_auc_values["oa_mean"] = mean_distance

    return sample_auc_values


def get_cuboid_corners(cuboids):
    M = cuboids.size(0)

    initial_corners = torch.tensor(
        [[[1., 1., 1.],
         [1., 1., -1.],
         [1., -1., 1.],
         [1., -1., -1.],
         [-1., 1., 1.],
         [-1., 1., -1.],
         [-1., -1., 1.],
         [-1., -1., -1.]]]
    ).expand(M, 8, 3).to(cuboids.device)

    corners = initial_corners * cuboids[:, :3].unsqueeze(1)

    angle_axis = cuboids[:, 3:6]
    t = cuboids[:, 6:9]

    R = pytorch3d.transforms.axis_angle_to_matrix(angle_axis)

    corners = corners @ R + t.unsqueeze(1)

    return corners


def get_points_in_hull(hull_points, size=(480, 640)):
    W, H = size

    coords_2d = np.stack(np.meshgrid(range(W), range(H)), axis=-1).reshape((-1, 2))

    M, Y, _ = hull_points.size()

    hull_choices = np.zeros((M, W, H), dtype=int)
    # hull_choices = torch.ones((P, S, K, B, W*H), device=choices.device, dtype=torch.int64)

    for mi in range(M):
        points = hull_points[mi]
        points = points.detach().cpu().numpy()
        try:
            # raise Exception
            triang = scipy.spatial.Delaunay(points)
            indices_in_hull = np.asarray(triang.find_simplex(coords_2d) >= 0).nonzero()
            hull_choices[mi][coords_2d[indices_in_hull][:,0], coords_2d[indices_in_hull][:,1]] = 1
            del triang
        except Exception as e:
            print(e)

    return hull_choices

def calc_cuboid_coverage(cuboids, image_size, K):

    corners_3d = get_cuboid_corners(cuboids)
    corners_3d[..., 2] = torch.clamp(corners_3d[..., 2], min=1e-3)
    corners = corners_3d @ K.transpose(-1, -2).to(corners_3d.device)
    corners = corners / corners[:, :, 2].unsqueeze(-1)
    corners = corners[:, :, :2]

    coverage = get_points_in_hull(corners, image_size)

    coverage = np.max(coverage, 0)

    return coverage


def calc_sq_coverage(visible_points, image_size, K):

    # visible_points_torch = torch.from_numpy(visible_points)
    K_torch = torch.from_numpy(K)
    corners = visible_points @ K_torch.transpose(-1, -2)
    corners = corners.unsqueeze(0)
    corners = corners / corners[:, :, 2].unsqueeze(-1)
    corners = corners[:, :, :2]

    coverage = get_points_in_hull(corners, image_size)

    coverage = np.max(coverage, 0)

    return coverage


def sq_io_fun(params, points, a_max=10., a_min=0.001):

    params = torch.where(params != params, torch.zeros_like(params), params)

    Y = points.size(0)

    a1 = torch.clamp(params[0], a_min, a_max).unsqueeze(-1)
    a2 = torch.clamp(params[1], a_min, a_max).unsqueeze(-1)
    a3 = torch.clamp(params[2], a_min, a_max).unsqueeze(-1)
    e1 = torch.clamp(params[3], 0.1, 1.9).unsqueeze(-1)
    e2 = torch.clamp(params[4], 0.1, 1.9).unsqueeze(-1)
    q = params[5:8]
    t = params[8:11]

    angle_axis = q
    # R = tgm.angle_axis_to_rotation_matrix(angle_axis.unsqueeze(0))
    R = p3d.transforms.axis_angle_to_matrix(angle_axis.unsqueeze(0))
    R = R[0, :3, :3]

    t_ = t.unsqueeze(0)
    points_ = points[:, :3]
    points_transformed = points_ - t_
    points_transformed = torch.matmul(points_transformed, R.transpose(-2, -1))
    del points_

    x = points_transformed[:, 0]
    y = points_transformed[:, 1]
    z = points_transformed[:, 2]

    x = torch.clamp(x/a1, -3, 3.)
    y = torch.clamp(y/a2, -3, 3.)
    z = torch.clamp(z/a3, -3, 3.)
    del points_transformed

    t1 = torch.clamp(x**2., min=1e-8, max=torch.finfo(torch.float32).max)
    t2 = torch.clamp(y**2., min=1e-8, max=torch.finfo(torch.float32).max)
    t3 = torch.clamp(z**2., min=1e-8, max=torch.finfo(torch.float32).max)
    del x, y, z

    t4 = torch.clamp(t1**(1/e2), min=1e-8, max=torch.finfo(torch.float32).max)
    t5 = torch.clamp(t2**(1/e2), min=1e-8, max=torch.finfo(torch.float32).max)
    del t1, t2

    u2 = t3**(1/e1)
    del t3
    u1 = torch.clamp(t4 + t5, min=1e-8, max=torch.finfo(torch.float32).max)**(e2/e1)
    del t4, t5

    F = u1 + u2

    return F-1