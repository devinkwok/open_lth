# callback for example difficulty metrics
import json
from itertools import chain
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn

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
        batch_size: int = None,
        verbose: bool = True,
    ):
        # make copy to avoid changing original, do not augment
        data_hparams = deepcopy(data_hparams)
        data_hparams.do_not_augment = True
        data_hparams.subset_end = n_examples
        if batch_size is None:
            batch_size = data_hparams.batch_size
        self.dataloader = get_dataloader(data_hparams, n_examples=n_examples, train=train, batch_size=batch_size)

        # need to init Accumulator objects before super().__init__() calls load()
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        metadata = {"device": get_device(), "n_examples": n_examples, "train": train, **data_hparams.to_dict()}
        self.pointwise_metrics = metrics.create_online_pointwise_metrics(**metadata)
        self.forget_metrics = metrics.create_online_forget_metrics(
            n_items=len(self.dataloader.dataset), start_at_zero=True, **metadata)
        self.other_metrics = {
            "avgloss": metrics.OnlineVariance(**metadata),
            "classvog": metrics.OnlineVarianceOfGradients(**metadata),
            "lossvog": metrics.OnlineVarianceOfGradients(loss_fn=self.loss_fn, **metadata),
        }

        super().__init__(schedule, output_location, iterations_per_epoch, verbose)

        # need to call super().__init__() before file operations
        hparam_file = self.callback_file("dataset.json")
        with get_platform().open(hparam_file, 'w') as fp:
            json.dump(data_hparams.to_dict(), fp, indent=4)

    def load_online_metrics(self, dict_of_metrics):
        output = {}
        for name, obj in dict_of_metrics.items():
            finished_it = set(i.iteration for i in self.finished_steps)
            file = self.callback_file(name)
            if not file.exists():
                raise RuntimeError(f"Unable to load {file}")
            output[name] = obj.load(file)
            # check that saved iterations are same as self.finished_steps
            it = set(output[name].get_metadata_lists("iteration"))
            if it != finished_it:
                union_over_intersection = (finished_it.union(it)).difference(finished_it.intersection(it))
                raise RuntimeError(f"Finished steps differ between {file} and {self.log_file}: {union_over_intersection}")
        return output

    def load(self):
        self.pointwise_metrics = self.load_online_metrics(self.pointwise_metrics)
        self.forget_metrics = self.load_online_metrics(self.forget_metrics)
        self.other_metrics = self.load_online_metrics(self.other_metrics)

    def save_checkpoint(self):
        for k, v in chain(self.pointwise_metrics.items(),
                          self.forget_metrics.items(),
                          self.other_metrics.items()):
            v.save(self.callback_file(k))

    def callback_function(self, output_location, step, model, optimizer, logger, *args, **kwds):
        metadata = {"iteration": step.iteration, "ep": step.ep, "it": step.it}
        # update vog metrics
        self.other_metrics["classvog"].add(model, self.dataloader, **metadata)
        _, logits = self.other_metrics["lossvog"].add(model, self.dataloader, return_output=True, **metadata)
        # compute loss
        labels = torch.cat([y for _, y in self.dataloader]).to(device=logits.device)
        loss = self.loss_fn(logits, labels).detach().cpu().to(dtype=torch.float64)
        self.other_metrics["avgloss"].add(loss, **metadata)
        # save loss at every scheduled step
        np.savez(self.callback_file("loss", step), loss)
        # update pointwise metrics
        pointwise = metrics.pointwise_metrics(logits, labels)
        acc = pointwise["acc"]
        for k, v in pointwise.items():
            self.pointwise_metrics[k].add(v, **metadata)
        # update forget metrics
        for v in self.forget_metrics.values():
            v.add(acc, **metadata)

        # save get() from online metrics if schedule is done
        if step == self.schedule.last_step:
            self.save_final_values(step)

    def save_final_values(self, step):
        for k, v in self.pointwise_metrics.items():
            np.savez(self.callback_file(k, step),
                     mean=v.get_mean().detach().cpu().numpy(),
                     variance=v.get().detach().cpu().numpy())
        # per-pixel means aren't useful for variance of gradients
        for k, v in chain(self.forget_metrics.items(), self.other_metrics.items()):
            np.savez(self.callback_file(k, step), v.get().detach().cpu().numpy())


    @staticmethod
    def name() -> str: return "pwmetrics"

    @staticmethod
    def description() -> str: return "Computes difficulty metrics throughout training on a fixed dataset."
