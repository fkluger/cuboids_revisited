import torch
import util.sampling


def manual_jacobian_x(y, x, create_graph=False):

    B = y.size(0)
    num_out = y.size(1)
    num_points = x.size(1)

    jac = torch.zeros((B, num_out, num_points, 3)).to(y.device)
    flat_y = y.view(B, -1)
    grad_y = torch.zeros_like(flat_y).to(flat_y.device)
    for i in range(num_out):
        grad_y[:, i] = 1.
        grad_x, = torch.autograd.grad(y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac[:, i] = grad_x
        grad_y[:, i] = 0.
    del grad_y
    return jac


def manual_jacobian_p(y, x, create_graph=False):

    B = y.size(0)
    num_out = y.size(1)
    num_points = x.size(1)

    jac = torch.zeros((B, num_out, num_points)).to(y.device)
    flat_y = y.view(B, -1)
    grad_y = torch.zeros_like(flat_y).to(flat_y.device)
    for i in range(num_out):
        grad_y[:, i] = 1.
        grad_x, = torch.autograd.grad(flat_y[:, i], x, grad_y[:, i], retain_graph=True, create_graph=create_graph)
        jac[:, i] = grad_x
        grad_y[:, i] = 0.
    del grad_y
    return jac


class CuboidWithImplicitFunGradient(torch.autograd.Function):

    def __init__(ctx):
        super().__init__()

    @staticmethod
    def forward(ctx, sq_params, points):
        if sq_params.requires_grad:
            sq_params.retain_grad()
        if points.requires_grad:
            points.retain_grad()
        ctx.save_for_backward(sq_params, points)
        return sq_params

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: dL/dp
        """

        dLdp = grad_output.unsqueeze(1).unsqueeze(1) # Bx1x1x9
        p, x = ctx.saved_tensors
        # p: sq_params, Bx9
        # x: points, Bx9x3
        with torch.enable_grad():
            p_ = p.detach()
            p_.requires_grad_()
            x_ = x.detach()
            x_.requires_grad_()
            F = util.sampling.cuboid_fun_weighted_torch(p_, x_, norm_by_volume=True)
            dFdx = manual_jacobian_x(F, x_).transpose(-1, -2).transpose(1, 2)
            dFdp = manual_jacobian_p(F, p_).unsqueeze(1)

        dFdp_ = dFdp[:, :, :, 3:]
        dFdp_inv = torch.pinverse(dFdp_)
        dpdx = torch.matmul(-dFdp_inv, dFdx)
        grad_input = torch.matmul(dLdp[:, :, :, 3:], dpdx)
        grad_input = grad_input.transpose(1, 2).transpose(2, 3).squeeze(1)

        del dLdp, p, x, p_, x_, dFdp, dFdx, dFdp_inv, dpdx, dFdp_

        return None, grad_input

