import torch
import torch.nn as nn
import torch.nn.functional as F
# import momentumnet


class Network(nn.Module):
    '''
    FCN architecture for neural-guided scene coordiante regression.
    The network has two output heads: One predicting a 3d scene coordinate, and a 1d neural guidance weight (log probability).
    The network makes dense predictions, but the output is subsampled by a factor of 8 compared to the input.
    '''

    OUTPUT_SUBSAMPLE = 8

    def __init__(self, data_channels=2, feature_size=0, instance_norm=True, num_probs=1, separate_probs=1,
                 bn_on_input=False, additional_prob=False):
        '''
        Constructor.
        '''
        super(Network, self).__init__()

        self.instance_norm = instance_norm

        self.feature_size = feature_size
        self.num_probs = num_probs
        self.separate_probs = separate_probs
        self.bn_on_input = bn_on_input
        self.additional_prob = additional_prob

        if self.bn_on_input:
            self.input_bn = nn.BatchNorm2d(data_channels)

        self.conv1 = nn.Conv2d(data_channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)

        self.res1_conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.res1_conv2 = nn.Conv2d(256, 256, 1, 1, 0)
        self.res1_conv3 = nn.Conv2d(256, 256, 3, 1, 1)

        self.res2_conv1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.res2_conv2 = nn.Conv2d(512, 512, 1, 1, 0)
        self.res2_conv3 = nn.Conv2d(512, 512, 3, 1, 1)

        self.res2_skip = nn.Conv2d(256, 512, 1, 1, 0)

        self.res3_conv1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.res3_conv2 = nn.Conv2d(512, 512, 1, 1, 0)
        self.res3_conv3 = nn.Conv2d(512, 512, 3, 1, 1)

        # output head 1, scene coordinates
        if self.feature_size > 0:
            self.fc1 = nn.Conv2d(512, 512, 1, 1, 0)
            self.fc2 = nn.Conv2d(512, 512, 1, 1, 0)
            self.fc3 = nn.Conv2d(512, self.feature_size, 1, 1, 0)

        # output head 2, neural guidance
        num_probs_ = self.num_probs+1 if self.additional_prob else self.num_probs
        self.fc1_1 = nn.Conv2d(512, 512, 1, 1, 0)
        self.fc2_1 = nn.Conv2d(512, 512, 1, 1, 0)
        self.fc3_1 = nn.Conv2d(512, num_probs_*self.separate_probs, 1, 1, 0)

        if self.num_probs > 1:
            self.q_out = nn.Conv2d(512, self.num_probs, 1, 1, 0)


    def forward(self, inputs):
        '''
        Forward pass.
        inputs -- 4D data tensor (BxCxHxW)
        '''

        batch_size = inputs.size(0)
        if self.bn_on_input:
            x = self.input_bn(inputs)
        else:
            x = inputs

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        if self.instance_norm:
            res = F.relu(F.instance_norm(self.conv4(x)))
        else:
            res = F.relu(self.conv4(x))

        x = F.relu(self.res1_conv1(res))
        x = F.relu(self.res1_conv2(x))
        if self.instance_norm:
            x = F.relu(F.instance_norm(self.res1_conv3(x)))
        else:
            x = F.relu(self.res1_conv3(x))

        res = res + x

        x = F.relu(self.res2_conv1(res))
        x = F.relu(self.res2_conv2(x))
        if self.instance_norm:
            x = F.relu(F.instance_norm(self.res2_conv3(x)))
        else:
            x = F.relu(self.res2_conv3(x))

        res = self.res2_skip(res) + x

        x = F.relu(self.res3_conv1(res))
        x = F.relu(self.res3_conv2(x))
        if self.instance_norm:
            x = F.relu(F.instance_norm(self.res3_conv3(x)))
        else:
            x = F.relu(self.res3_conv3(x))

        res = res + x

        # output head 1, scene coordinates
        if self.feature_size > 0:
            feat = F.relu(self.fc1(res))
            feat = F.relu(self.fc2(feat))
            feat = self.fc3(feat)

        # output head 2, neural guidance
        log_ng_1 = F.relu(self.fc1_1(res))
        log_ng_2 = F.relu(self.fc2_1(log_ng_1))
        log_ng = self.fc3_1(log_ng_2)

        # normalize neural guidance probabilities in log space
        log_ng = F.logsigmoid(log_ng)
        log_ng_ = log_ng.view(batch_size, -1)

        normalizer = torch.logsumexp(log_ng_, dim=1, keepdim=True)
        log_ng_ = log_ng_ - normalizer
        log_ng = log_ng_.view(log_ng.size())

        if self.num_probs > 1:
            y = F.adaptive_avg_pool2d(log_ng_2, (1, 1))
            log_q = F.logsigmoid(self.q_out(y)).squeeze()
        else:
            log_q = None

        if self.feature_size > 0:
            return log_ng, feat, log_q
        else:
            return log_ng, log_q


class SmallNetwork(nn.Module):
    '''
    FCN architecture for neural-guided scene coordiante regression.
    The network has two output heads: One predicting a 3d scene coordinate, and a 1d neural guidance weight (log probability).
    The network makes dense predictions, but the output is subsampled by a factor of 8 compared to the input.
    '''

    OUTPUT_SUBSAMPLE = 8

    def __init__(self, data_channels=2, feature_size=0, instance_norm=True, num_probs=1, separate_probs=1,
                 bn_on_input=False, additional_prob=False):
        '''
        Constructor.
        '''
        super(SmallNetwork, self).__init__()

        self.instance_norm = instance_norm

        self.feature_size = feature_size
        self.num_probs = num_probs
        self.separate_probs = separate_probs
        self.bn_on_input = bn_on_input
        self.additional_prob = additional_prob

        if self.bn_on_input:
            self.input_bn = nn.BatchNorm2d(data_channels)

        self.conv1 = nn.Conv2d(data_channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)

        self.res1_conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.res1_conv2 = nn.Conv2d(256, 256, 1, 1, 0)
        self.res1_conv3 = nn.Conv2d(256, 256, 3, 1, 1)

        self.res2_conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.res2_conv2 = nn.Conv2d(256, 256, 1, 1, 0)
        self.res2_conv3 = nn.Conv2d(256, 256, 3, 1, 1)

        self.res2_skip = nn.Conv2d(256, 256, 1, 1, 0)

        self.res3_conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.res3_conv2 = nn.Conv2d(256, 256, 1, 1, 0)
        self.res3_conv3 = nn.Conv2d(256, 256, 3, 1, 1)

        # output head 2, neural guidance
        num_probs_ = self.num_probs+1 if self.additional_prob else self.num_probs
        self.fc1_1 = nn.Conv2d(256, 256, 1, 1, 0)
        self.fc2_1 = nn.Conv2d(256, 256, 1, 1, 0)
        self.fc3_1 = nn.Conv2d(256, num_probs_*self.separate_probs, 1, 1, 0)

        if self.num_probs > 1:
            self.q_out = nn.Conv2d(256, self.num_probs, 1, 1, 0)


    def forward(self, inputs):
        '''
        Forward pass.
        inputs -- 4D data tensor (BxCxHxW)
        '''

        batch_size = inputs.size(0)
        if self.bn_on_input:
            x = self.input_bn(inputs)
        else:
            x = inputs

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        if self.instance_norm:
            res = F.relu(F.instance_norm(self.conv4(x)))
        else:
            res = F.relu(self.conv4(x))

        x = F.relu(self.res1_conv1(res))
        x = F.relu(self.res1_conv2(x))
        if self.instance_norm:
            x = F.relu(F.instance_norm(self.res1_conv3(x)))
        else:
            x = F.relu(self.res1_conv3(x))

        res = res + x

        x = F.relu(self.res2_conv1(res))
        x = F.relu(self.res2_conv2(x))
        if self.instance_norm:
            x = F.relu(F.instance_norm(self.res2_conv3(x)))
        else:
            x = F.relu(self.res2_conv3(x))

        res = res + x

        x = F.relu(self.res3_conv1(res))
        x = F.relu(self.res3_conv2(x))
        if self.instance_norm:
            x = F.relu(F.instance_norm(self.res3_conv3(x)))
        else:
            x = F.relu(self.res3_conv3(x))

        res = res + x

        # output head 1, scene coordinates
        if self.feature_size > 0:
            feat = F.relu(self.fc1(res))
            feat = F.relu(self.fc2(feat))
            feat = self.fc3(feat)

        # output head 2, neural guidance
        log_ng_1 = F.relu(self.fc1_1(res))
        log_ng_2 = F.relu(self.fc2_1(log_ng_1))
        log_ng = self.fc3_1(log_ng_2)

        # normalize neural guidance probabilities in log space
        log_ng = F.logsigmoid(log_ng)
        log_ng_ = log_ng.view(batch_size, -1)

        normalizer = torch.logsumexp(log_ng_, dim=1, keepdim=True)
        log_ng_ = log_ng_ - normalizer
        log_ng = log_ng_.view(log_ng.size())

        if self.num_probs > 1:
            y = F.adaptive_avg_pool2d(log_ng_2, (1, 1))
            log_q = F.logsigmoid(self.q_out(y)).squeeze()
        else:
            log_q = None

        if self.feature_size > 0:
            return log_ng, feat, log_q
        else:
            return log_ng, log_q

