# callback for example difficulty metrics
from typing import Any
import torch

from foundations.hparams import DatasetHparams
from training.callbacks import base
from training.utils import batch_seed
from api import get_device, get_dataset
from datasets.base import ShuffleSampler

import sys
sys.path.append("nn_example_difficulty")
from difficulty import metrics
from difficulty.model.eval import evaluate_model


class Callback(base.Callback):
    def __init__(self, output_location, iterations_per_epoch, data_hparams: DatasetHparams, verbose: bool = True):
        self.batch_size = data_hparams.batch_size
        # do not augment data, get separate dataset to avoid affecting training loop
        self.data = get_dataset(data_hparams).get_train_set(False)
        self.sampler = ShuffleSampler(len(self.data))
        self.batch_forget = metrics.OnlineCountForgetting(
                n_items=len(self.data), device=get_device())
        super().__init__(base.CallbackSchedule(None), output_location, iterations_per_epoch, verbose)

    def load(self, output_location):
        finished_it = set(i.iteration for i in self.finished_steps)
        file = output_location / self.metric_filename("nforget.npz")
        if not file.exists():
            raise RuntimeError(f"Unable to load {file}")
        self.batch_forget = self.batch_forget.load(file)
        # check that metadata["iteration"] is same as self.finished_steps
        it = set(self.batch_forget.metadata["iteration"])
        if it != finished_it:
            union_over_intersection = (finished_it.union(it)).difference(finished_it.intersection(it))
            raise RuntimeError(f"Finished steps differ between {file} and {self.log_file}: {union_over_intersection}")

    def save(self, output_location):
        self.batch_forget.save(output_location / self.metric_filename("count_forgetting.npz"))

    def minibatch_signature(labels):
        return hash(tuple(labels.detach().cpu().numpy()))

    def callback_function(self, output_location, step, model, optimizer, logger, examples, labels) -> Any:
        metadata = {"iteration": step.iteration, "ep": step.ep, "it": step.it}

        with base.ExecTime("Batch forgetting", verbose=self.verbose):
            minibatch_idx, batch, batch_metadata = self._get_minibatch_idx(step, labels)
            _, _, minibatch_acc, _ = evaluate_model(model, batch, device=get_device(), return_accuracy=True)
            self.metrics["batch-forget"].add(minibatch_acc, minibatch_idx=minibatch_idx,
                                             **batch_metadata, **metadata)

    def _get_minibatch_idx(self, step, labels):
        seed = batch_seed(self.training_hparams, step.ep)
        shuffle_idx = list(iter(self.sampler.shuffle_dataorder(seed)))
        start = step.it * self.batch_size
        minibatch_idx = shuffle_idx[start:start + self.batch_size]
        batch_data = self.data[minibatch_idx]
        batch_labels = torch.tensor([y for _, y  in batch_data], dtype=torch.long)
        # check that batches match
        assert torch.equal(labels, batch_labels)
        signature = self.minibatch_signature(labels)
        return minibatch_idx, [(batch_data, batch_labels)], {"seed": seed, "signature": signature}

    def name() -> str: "batchforget"

    def description() -> str: "Computes forgetting metrics for training batches."
