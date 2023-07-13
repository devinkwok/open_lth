# callback for example difficulty metrics
import json
from copy import deepcopy
from pathlib import Path
import hashlib
import torch.nn as nn

from foundations.hparams import DatasetHparams
from platforms.platform import get_platform
from training.callbacks import base
from api import get_device, get_dataloader

import sys
sys.path.append("nn_example_difficulty")
from difficulty import metrics
from difficulty.model.eval import evaluate_model


class Callback(base.Callback):
    def __init__(self,
        data_hparams: DatasetHparams,
        n_examples: int,
        train: bool,
        schedule: base.CallbackSchedule,
        output_location,
        iterations_per_epoch,
        batch_size: int = None,
        grad_batch_size: int = None,
        use_functional_grad: bool = True,
        verbose: bool = True,
    ):
        # make copy to avoid changing original, do not augment
        data_hparams = deepcopy(data_hparams)
        data_hparams.do_not_augment = True
        data_hparams.subset_end = n_examples
        if grad_batch_size is None:
            grad_batch_size = data_hparams.batch_size
        self.dataloader = get_dataloader(data_hparams, n_examples=n_examples, train=train, batch_size=batch_size)
        self.grad_dataloader = get_dataloader(data_hparams, n_examples=n_examples, train=train, batch_size=grad_batch_size)
        self.use_functional_grad = use_functional_grad
        # identifier based on dataloader hparams, only 10 hash digits as it's unlikely to collide
        self.hash = hashlib.md5(str(data_hparams).encode('utf-8')).hexdigest()[:10] + ("train" if train else "test")

        # need to init Accumulator objects before super().__init__() calls load()
        self.metrics = {
            "loss": metrics.OnlineVariance(device=get_device()),
            "acc": metrics.OnlineVariance(device=get_device()),
            "ent": metrics.OnlineVariance(device=get_device()),
            "conf": metrics.OnlineVariance(device=get_device()),
            "maxconf": metrics.OnlineVariance(device=get_device()),
            "margin": metrics.OnlineVariance(device=get_device()),
            "el2n": metrics.OnlineVariance(device=get_device()),
            "nforget": metrics.OnlineCountForgetting(
                n_items=len(self.dataloader.dataset), device=get_device()),
            "grand": metrics.OnlineVariance(device=get_device()),
            "vog": metrics.OnlineVarianceOfGradients(device=get_device()),
        }

        super().__init__(schedule, output_location, iterations_per_epoch, verbose)

        # need to call super().__init__() before file operations
        hparam_file = self.callback_file("dataset.json")
        with get_platform().open(hparam_file, 'w') as fp:
            json.dump(data_hparams.to_dict(), fp, indent=4)

    def load(self):
        finished_it = set(i.iteration for i in self.finished_steps)
        for k, v in self.metrics.items():
            file = self.callback_file(k)
            if not file.exists():
                raise RuntimeError(f"Unable to load {file}")
            self.metrics[k] = v.load(file)
            # check that metadata["iteration"] is same as self.finished_steps
            it = set(self.metrics[k].metadata["iteration"])
            if it != finished_it:
                union_over_intersection = (finished_it.union(it)).difference(finished_it.intersection(it))
                raise RuntimeError(f"Finished steps differ between {file} and {self.log_file}: {union_over_intersection}")

    def save(self):
        for k, v in self.metrics.items():
            v.save(self.callback_file(k))

    def callback_function(self, output_location, step, model, optimizer, logger, *args, **kwds):
        metadata = {"iteration": step.iteration, "ep": step.ep, "it": step.it}

        with base.ExecTime(f"{self.name()} {self.hash()}", verbose=self.verbose):
            logits, labels, acc, loss = evaluate_model(
                model, self.dataloader, device=get_device(),
                loss_fn=nn.CrossEntropyLoss(reduction="none")
            )
            self.metrics["loss"].add(loss, **metadata)
            for k, v in self.pointwise_metrics(logits, labels).items():
                self.metrics[k].add(v, **metadata)
            self.metrics["nforget"].add(acc, **metadata)

        with base.ExecTime("GraNd score", verbose=self.verbose):
            self.metrics["grand"].add(metrics.grand_score(
                model, self.grad_dataloader, device=get_device(), use_functional=self.use_functional_grad), **metadata)

        with base.ExecTime("VoG metric", verbose=self.verbose):
            self.metrics["vog"].add(model, self.grad_dataloader, **metadata)

    @staticmethod
    def pointwise_metrics(eval_logits, labels):
        prob = metrics.softmax(eval_logits)
        return {
            "acc": metrics.zero_one_accuracy(eval_logits, labels),
            "ent": metrics.entropy(eval_logits, labels),
            "conf": metrics.class_confidence(prob, labels),
            "maxconf": metrics.max_confidence(prob),
            "margin": metrics.margin(prob, labels),
            "el2n": metrics.error_l2_norm(prob, labels),
        }

    @staticmethod
    def name() -> str: return "pwmetrics"

    @staticmethod
    def description() -> str: return "Computes difficulty metrics throughout training on a fixed dataset."

    @property
    def save_root(self): return self.output_location / f"{self.name()}_{self.hash}"

    def callback_file(self, key: str):
        if "." not in key:  # add .npz suffix by default
            key = f"{key}.npz"
        return super().callback_file(key)
