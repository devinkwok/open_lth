# callback for example difficulty metrics
import json
from copy import deepcopy
import numpy as np
import torch

from foundations.hparams import DatasetHparams
from platforms.platform import get_platform
from training.callbacks import base 
from api import get_device, get_dataloader

import sys
sys.path.append("nn_example_difficulty")
from difficulty import metrics


class Callback(base.Callback):
    def __init__(self,
        data_hparams: DatasetHparams,
        n_examples: int,
        train: bool,
        schedule: base.CallbackSchedule,
        output_location,
        iterations_per_epoch,
        batch_size: int = 40,
        use_functional_grad: bool = True,
        verbose: bool = True,
    ):
        super().__init__(schedule, output_location, iterations_per_epoch, verbose)

        # make copy to avoid changing original, do not augment
        data_hparams = deepcopy(data_hparams)
        data_hparams.do_not_augment = True
        data_hparams.subset_end = n_examples
        if batch_size is None:
            batch_size = data_hparams.batch_size
        self.dataloader = get_dataloader(data_hparams, n_examples=n_examples, train=train, batch_size=batch_size)
        self.use_functional_grad = use_functional_grad
        self.dtype = torch.float64

        # need to call super().__init__() before file operations
        hparam_file = self.callback_file("dataset.json")
        with get_platform().open(hparam_file, 'w') as fp:
            json.dump(data_hparams.to_dict(), fp, indent=4)

    def load(self):
        pass  # no state to load

    def save_checkpoint(self):
        pass  # no state to save

    def callback_function(self, output_location, step, model, optimizer, logger, *args, **kwds):
        with base.ExecTime(f"EL2N and GraNd", verbose=self.verbose):
            # get GraNd score
            grand, eval_logits = metrics.grand_score(model, self.dataloader, device=get_device(),
                                use_functional=self.use_functional_grad, return_output=True)
            np.savez(self.callback_file("grand", step), grand=grand.detach().cpu().numpy())
            # get EL2N metric
            prob = metrics.softmax(eval_logits.to(self.dtype))
            labels = torch.cat([y for _, y in self.dataloader]).to(device=prob.device)
            el2n = metrics.error_l2_norm(prob, labels)
            np.savez(self.callback_file("el2n", step), el2n=el2n.detach().cpu().numpy())

    @staticmethod
    def name() -> str: return "gradmetrics"

    @staticmethod
    def description() -> str: return "Computes EL2N and GraNd metrics at specific iterations on a fixed dataset."
