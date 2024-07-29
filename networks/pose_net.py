import torch.nn as nn
import torch.nn.functional as F
from util.consistency import *
import pytorch3d.transforms


def twist_to_matrix(moment, omega, theta):

    B = moment.size(0)

    matrix = torch.zeros(B, 4, 4, device=moment.device)

    omega_hat = torch.zeros(B, 3, 3, device=moment.device)

    omega_hat[:, 0, 1] = -omega[:, 2]
    omega_hat[:, 0, 2] = omega[:, 1]
    omega_hat[:, 1, 0] = omega[:, 2]
    omega_hat[:, 1, 2] = -omega[:, 0]
    omega_hat[:, 2, 0] = -omega[:, 1]
    omega_hat[:, 2, 1] = omega[:, 0]

    omega_t = omega.unsqueeze(2) @ omega.unsqueeze(1)

    R = (omega_hat * torch.sin(theta).unsqueeze(1))
    R += ((omega_hat @ omega_hat) * (1-torch.cos(theta)).unsqueeze(1))
    R += torch.eye(3).unsqueeze(0).to(moment.device)

    t1 = (torch.eye(3).unsqueeze(0).to(moment.device)-R)
    t2 = torch.cross(omega, moment, dim=-1)
    t = t1 @ t2.unsqueeze(-1)
    t += omega_t @ moment.unsqueeze(-1) * theta.unsqueeze(1)

    matrix[:, 0:3, 0:3] = R
    matrix[:, 0:3, 3] = t.squeeze(-1)
    matrix[:, 3, 3] += 1

    return matrix

class TransformerPoseNet(nn.Module):
    '''
    Reimplementation of the network "Learning to find good correspondences"

    '''

    def __init__(self, tf_layers, input_dim, bias=True, iterations=1, a_min=0.01, a_max=2.):
        '''
        Constructor.
        '''
        super(TransformerPoseNet, self).__init__()

        self.input_dim = input_dim
        self.iterations = iterations

        self.p_in = nn.Conv2d(self.input_dim, 128, 1, 1, 0, bias=bias)

        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(128, 128, bias=bias)
        self.fc2 = nn.Linear(128, 128, bias=bias)

        self.fc_rotation = nn.Linear(128, 6, bias=bias)
        self.fc_translation = nn.Linear(128, 3, bias=bias)
        self.fc_size = nn.Linear(128, 3, bias=bias)

        self.a_max = a_max
        self.a_min = a_min

    def forward(self, inputs):
        '''
        Forward pass.

        inputs -- 4D data tensor (BxCxHxW)
        '''
        inputs_ = torch.transpose(inputs, 1, 2).unsqueeze(-1)

        x = inputs_[:, 0:self.input_dim]
        x = F.relu(self.p_in(x))

        x = torch.transpose(x, 1, 2).squeeze(-1)
        x = self.transformer_encoder(x)
        x = x.unsqueeze(-1)
        x = torch.transpose(x, 1, 2)

        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        rot_6d = torch.tanh(self.fc_rotation(x))
        R = pytorch3d.transforms.rotation_6d_to_matrix(rot_6d)

        t = torch.tanh(self.fc_translation(x))

        size = torch.sigmoid(self.fc_size(x))

        quaternion = pytorch3d.transforms.matrix_to_quaternion(R)

        axis_angle = pytorch3d.transforms.quaternion_to_axis_angle(quaternion)


        return size, axis_angle, t
