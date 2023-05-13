# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import torch
import typing

from foundations import paths
from foundations.step import Step
import lottery.desc
from platforms.platform import get_platform


class OutputAffineLayerNorm(torch.nn.Module):
    """IMPORTANT NOTE
    This replicates [Ainsworth, Hayase, Srinivasa. Git re-basin: Merging models modulo permutation symmetries]
    Specifically, mean and std are computed over the output channels only,
    rather than also over image and input dimensions H, W, C.
    """

    def __init__(self, num_features: int, layernorm_scaling=1):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(num_features)
        self.scaling = layernorm_scaling  #TODO temporary hack for layernorm

    def forward(self, x):
        if len(x.shape) > 2:  # for conv layers, there are additional dims after output dim
            x = torch.moveaxis(x, 1, -1)  # move output dim to end
            original_shape = x.shape
            x = x.reshape(-1, original_shape[-1])
            x = self._custom_layernorm(x)
            x = x.reshape(original_shape)  # go back to original shape
            x = torch.moveaxis(x, -1, 1)
        else:
            x = self._custom_layernorm(x)
        return x

    def _custom_layernorm(self, x):
        if self.scaling == 1:
            return self.layernorm(x)
        # TODO temporary hack for inference only: makes layernorm work with padded networks
        std, padded_mean = torch.std_mean(x, dim=-1, unbiased=False)  # nn.LayerNorm uses biased variance calculation
        if self.scaling == 0:  # ignore mean and std if scaling = 0
            return x * self.layernorm.weight + self.layernorm.bias
        # correct for extra zeros from padding, e.g. if padded 2x, then mean should be 2x
        mean = (padded_mean * self.scaling).reshape(-1, 1)
        var = self.scaling * (std**2 + (1 - self.scaling) * padded_mean**2).reshape(-1, 1)
        # replicate layernorm
        normalized = (x - mean) / torch.sqrt(var + self.layernorm.eps) * self.layernorm.weight + self.layernorm.bias
        return normalized


class Model(torch.nn.Module, abc.ABC):
    """The base class used by all models in this codebase."""

    @staticmethod
    @abc.abstractmethod
    def is_valid_model_name(model_name: str) -> bool:
        """Is the model name string a valid name for models in this class?"""

        pass

    @staticmethod
    @abc.abstractmethod
    def get_model_from_name(
        model_name: str,
        outputs: int,
        initializer: typing.Callable[[torch.nn.Module], None]
    ) -> 'Model':
        """Returns an instance of this class as described by the model_name string."""

        pass

    @property
    def prunable_layer_names(self) -> typing.List[str]:
        """A list of the names of Tensors of this model that are valid for pruning.

        By default, only the weights of convolutional and linear layers are prunable.
        """

        return [name + '.weight' for name, module in self.named_modules() if
                isinstance(module, torch.nn.modules.conv.Conv2d) or
                isinstance(module, torch.nn.modules.linear.Linear)]

    @property
    @abc.abstractmethod
    def output_layer_names(self) -> typing.List[str]:
        """A list of the names of the Tensors of the output layer of this model."""

        pass

    @staticmethod
    @abc.abstractmethod
    def default_hparams() -> 'lottery.desc.LotteryDesc':
        """The default hyperparameters for training this model and running lottery ticket."""

        pass

    @property
    @abc.abstractmethod
    def loss_criterion(self) -> torch.nn.Module:
        """The loss criterion to use for this model."""

        pass

    def save(self, save_location: str, save_step: Step):
        if not get_platform().is_primary_process: return
        if not get_platform().exists(save_location): get_platform().makedirs(save_location)
        get_platform().save_model(self.state_dict(), paths.model(save_location, save_step))

    @staticmethod
    def get_batchnorm(n_filters, batchnorm_type=None):
        if batchnorm_type is None:
            return torch.nn.BatchNorm2d(n_filters)
        substrings = batchnorm_type.split("$")  #TODO temporary hack for layernorm
        batchnorm_type = substrings[0]
        layernorm_scaling = substrings[1] if len(substrings) > 1 else 1
        if batchnorm_type == "bn":
            return torch.nn.BatchNorm2d(n_filters)
        if batchnorm_type == "layernorm":
            return OutputAffineLayerNorm(n_filters, float(layernorm_scaling))
        if batchnorm_type == "linear":
            return torch.nn.Conv2d(n_filters, n_filters, kernel_size=1, bias=True)
        if batchnorm_type == "none-bias" or batchnorm_type == "none":
            return torch.nn.Identity(n_filters)
        raise ValueError(f"Batchnorm type {batchnorm_type} must be None, 'bn', 'layernorm', 'linear', 'none-bias', or 'none'.")

    @staticmethod
    def use_conv_bias(batchnorm_type):
        if batchnorm_type is None:
            return False
        return batchnorm_type.endswith("bias")


class DataParallel(Model, torch.nn.DataParallel):
    def __init__(self, module: Model):
        super(DataParallel, self).__init__(module=module)

    @property
    def prunable_layer_names(self): return self.module.prunable_layer_names

    @property
    def output_layer_names(self): return self.module.output_layer_names

    @property
    def loss_criterion(self): return self.module.loss_criterion

    @staticmethod
    def get_model_from_name(model_name, initializer, outputs, batchnorm_type):
        raise NotImplementedError

    @staticmethod
    def is_valid_model_name(model_name): raise NotImplementedError

    @staticmethod
    def default_hparams(): raise NotImplementedError

    def save(self, save_location: str, save_step: Step):
        self.module.save(save_location, save_step)


class DistributedDataParallel(Model, torch.nn.parallel.DistributedDataParallel):
    def __init__(self, module: Model, device_ids):
        super(DistributedDataParallel, self).__init__(module=module, device_ids=device_ids)

    @property
    def prunable_layer_names(self): return self.module.prunable_layer_names

    @property
    def output_layer_names(self): return self.module.output_layer_names

    @property
    def loss_criterion(self): return self.module.loss_criterion

    @staticmethod
    def get_model_from_name(model_name, outputs, initializer): raise NotImplementedError

    @staticmethod
    def is_valid_model_name(model_name): raise NotImplementedError

    @staticmethod
    def default_hparams(): raise NotImplementedError

    def save(self, save_location: str, save_step: Step):
        self.module.save(save_location, save_step)
