# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global


class Model(base.Model):
    """A VGG-style neural network designed for CIFAR-10."""

    class ConvModule(nn.Module):
        """A single convolutional module in a VGG network."""

        def __init__(self, in_filters, out_filters, batchnorm_type=None):
            super(Model.ConvModule, self).__init__()
            self.relu = nn.ReLU()
            self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
            self.bn = Model.get_batchnorm(out_filters, batchnorm_type)

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    def __init__(self, plan, initializer, outputs=10, batchnorm_type=None):
        super(Model, self).__init__()

        layers = []
        filters = 3

        for spec in plan:
            if spec == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(Model.ConvModule(filters, spec, batchnorm_type))
                filters = spec

        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(plan[-1], outputs)
        self.criterion = nn.CrossEntropyLoss()

        self.apply(initializer)

    def forward(self, x):
        x = self.layers(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('cifar_vgg_') and
                5 > len(model_name.split('_')) > 2 and
                all([x.isdigit() and int(x) > 0 for x in model_name.split('_')[2:]]) and
                int(model_name.split('_')[2]) in [11, 13, 16, 19])

    @staticmethod
    def hidden_numel_from_model_name(model_name):
        plan = Model.plan_from_model_name(model_name)
        ch, h, w = 3, 32, 32
        numel = ch * h * w
        for spec in plan:
            if spec == 'M':
                h, w, num_submodules = h/2, w/2, 1
            else:
                ch, num_submodules = spec, 3
            numel += ch * h * w * num_submodules
        # AvgPool
        numel += plan[-1]
        # FC
        numel += 10
        return numel

    @staticmethod
    def plan_from_model_name(model_name):

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        name = model_name.split('_')
        W = 64 if len(name) == 3 else int(name[3])
        num = int(name[2])
        if num == 11:
            plan = [W, 'M', 2*W, 'M', 4*W, 4*W, 'M', 8*W, 8*W, 'M', 8*W, 8*W]
        elif num == 13:
            plan = [W, W, 'M', 2*W, 2*W, 'M', 4*W, 4*W, 'M', 8*W, 8*W, 'M', 8*W, 8*W]
        elif num == 16:
            plan = [W, W, 'M', 2*W, 2*W, 'M', 4*W, 4*W, 4*W, 'M', 8*W, 8*W, 8*W, 'M', 8*W, 8*W, 8*W]
        elif num == 19:
            plan = [W, W, 'M', 2*W, 2*W, 'M', 4*W, 4*W, 4*W, 4*W, 'M', 8*W, 8*W, 8*W, 8*W, 'M', 8*W, 8*W, 8*W, 8*W]
        else:
            raise ValueError('Unknown VGG model: {}'.format(model_name))
        return plan

    @staticmethod
    def get_model_from_name(model_name, initializer, outputs=10, batchnorm_type=None):
        """The naming scheme for VGG is 'cifar_vgg_N[_W]'.
        N is number of layers, W is width.
        If W is not set, the default width is 64 in the first convolution layers
        and 512 in the last convolution layers.
        """
        plan = Model.plan_from_model_name(model_name)
        outputs = outputs or 10
        return Model(plan, initializer, outputs, batchnorm_type)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='cifar_vgg_16',
            model_init='kaiming_normal',
            batchnorm_init='uniform',
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='cifar10',
            batch_size=128
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='sgd',
            momentum=0.9,
            milestone_steps='80ep,120ep',
            lr=0.1,
            gamma=0.1,
            weight_decay=1e-4,
            training_steps='160ep'
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2,
            pruning_layers_to_ignore='fc.weight'
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)
