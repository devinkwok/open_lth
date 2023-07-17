# callback for example difficulty metrics
from typing import Any
import torch
import numpy as np

from foundations.step import Step
from foundations.hparams import TrainingHparams, DatasetHparams
from training.callbacks import base
from training.utils import batch_seed
from api import get_device, get_dataset
from datasets.base import ShuffleSampler

import sys
sys.path.append("nn_example_difficulty")
from difficulty import metrics
from difficulty.model.eval import evaluate_model


class Callback(base.Callback):
    def __init__(self,
                 output_location,
                 iterations_per_epoch,
                 train_hparams: TrainingHparams,
                 data_hparams: DatasetHparams,
                 start_at_zero: bool=True,
                 verbose: bool = True
    ):
        self.train_hparams = train_hparams
        self.last_step = Step.from_str(train_hparams.training_steps, iterations_per_epoch)
        if train_hparams.data_order_seed is None:
            raise ValueError("Must set TrainingHparams.data_order_seed")
        self.batch_size = data_hparams.batch_size
        # do not augment data, get separate dataset to avoid affecting training loop
        self.data = get_dataset(data_hparams).get_train_set(False)
        self.sampler = ShuffleSampler(len(self.data))
        # create metrics accumulator objects
        self.forget_metrics = metrics.create_online_forget_metrics(
            n_items=len(self.data), device=get_device(),
            start_at_zero=start_at_zero, **data_hparams.to_dict())
        # schedule is run on every iteration
        super().__init__(base.CallbackSchedule(None), output_location, iterations_per_epoch, verbose)

    def load(self):
        for name, obj in self.forget_metrics.items():
            finished_it = set(i.iteration for i in self.finished_steps)
            file = self.callback_file(name)
            if not file.exists():
                raise RuntimeError(f"Unable to load {file}")
            self.forget_metrics[name] = obj.load(file)
            # check that saved iterations are same as self.finished_steps
            it = set(self.forget_metrics[name].get_metadata_lists("iteration"))
            if it != finished_it:
                union_over_intersection = (finished_it.union(it)).difference(finished_it.intersection(it))
                raise RuntimeError(f"Finished steps differ between {file} and {self.log_file}: {union_over_intersection}")

    def save_checkpoint(self):
        for k, v in self.forget_metrics.items():
            v.save(self.callback_file(k))

    def callback_function(self, output_location, step, model, optimizer, logger, examples, labels) -> Any:
        metadata = {"iteration": step.iteration, "ep": step.ep, "it": step.it}
        minibatch_idx, batch, batch_metadata = self._get_minibatch_idx(step, labels)
        _, _, minibatch_acc, _ = evaluate_model(model, batch, device=get_device(), return_accuracy=True)
        for v in self.forget_metrics.values():
            v.add(minibatch_acc, minibatch_idx=minibatch_idx, **metadata, **batch_metadata)
        # save get() from online metrics if schedule is done
        if step == self.last_step:
            for k, v in self.forget_metrics.items():
                np.savez(self.callback_file(k, step), value=v.get().detach().cpu().numpy())

    def _get_minibatch_idx(self, step, labels):
        # get batch order
        seed = batch_seed(self.train_hparams, step.ep)
        self.sampler.shuffle_dataorder(seed)  # in place
        shuffle_idx = list(iter(self.sampler))
        start = step.it * self.batch_size
        # get batch data
        minibatch_idx = torch.tensor(shuffle_idx[start:start + self.batch_size], dtype=torch.long, device=get_device())
        batch_data, batch_labels = tuple(zip(*[self.data[i] for i in minibatch_idx]))
        batch_data = torch.stack(batch_data, dim=0)
        batch_labels = torch.tensor(batch_labels)
        # check that batches match
        assert torch.equal(labels, batch_labels), (labels, batch_labels)
        signature = self.minibatch_signature(labels)
        return minibatch_idx, [(batch_data, batch_labels)], {"seed": seed, "signature": signature}

    @staticmethod
    def minibatch_signature(labels):
        return hash(tuple(labels.detach().cpu().numpy()))

    @staticmethod
    def name() -> str: return "batchforget"

    @staticmethod
    def description() -> str: return "Computes forgetting metrics for training batches."
