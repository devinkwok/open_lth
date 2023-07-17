import abc
from time import time
from typing import Any
from pathlib import Path

from foundations.step import Step
from platforms.platform import get_platform
from training.metric_logger import MetricLogger

import time

class Stopwatch:
    def __init__(self, name="STOPWATCH"):
        self.name = name
        self.n_total = 0
        self.total_time = 0
        self.n_since_last_print = 0
        self.time_since_last_print = 0
        self.start_time = None
        self.stop_time = None

    def start(self):
        self.n_total += 1
        self.n_since_last_print += 1
        self.start_time = time.time()

    def stop(self):
        self.stop_time = time.time()
        elapsed = self.stop_time - self.start_time
        self.total_time += elapsed
        self.time_since_last_print += elapsed

    def print(self, message=""):
        elapsed = 0
        # if stop() hasn't been called, return current time minus start
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if self.stop_time is not None and self.stop_time > self.start_time:
                elapsed = self.stop_time - self.start_time
        if message != "":
            message = " " + message
        print(f"{self.name}{message}\telapsed {elapsed:0.2f}\tlast {self.time_since_last_print:0.2f} " +
              f"({self.n_since_last_print})\ttotal {self.total_time:0.2f} ({self.n_total})")
        self.n_since_last_print = 0
        self.time_since_last_print = 0


class CallbackSchedule:

    def __init__(self, steps=None) -> None:
        self.steps = steps

    @property
    def last_step(self):
        return max(self.steps)

    @staticmethod
    def from_str(string, iterations_per_epoch):
        steps = CallbackSchedule.iter_str_to_steps(string, iterations_per_epoch)
        return CallbackSchedule(steps)

    @staticmethod
    def iter_str_to_steps(iters: str, iterations_per_epoch: int):
        step_list = []
        elements = iters.split(',')
        for element in elements:
            if len(element.split('-')) == 2:  # range of steps
                start, end_skip = element.split('-')
                start = Step.from_str(start, iterations_per_epoch)
                if len(end_skip.split('@')) == 2:
                    end = Step.from_str(end_skip.split('@')[0], iterations_per_epoch)
                    skip = Step.from_str(end_skip.split('@')[1], iterations_per_epoch)
                else:  # if not set, assume skip is every 1 epoch
                    end = Step.from_str(end_skip, iterations_per_epoch)
                    skip = Step.one_ep(iterations_per_epoch)
                while start <= end:
                    step_list.append(start)
                    start = start + skip
            else:  # single step
                step_list.append(Step.from_str(element, iterations_per_epoch))
        return set(step_list)

    def do_run(self, step):
        if self.steps is None:
            return True
        return step in self.steps


class Callback:
    def __init__(self,
                 schedule: CallbackSchedule,
                 output_location,
                 iterations_per_epoch,
                 verbose: bool = True,
        ):
        self.schedule = schedule
        self.output_location = Path(output_location)
        self.iterations_per_epoch = iterations_per_epoch
        self.verbose = verbose
        # get finished steps
        self.logger = MetricLogger()
        self.last_save_ep = -1
        # load
        if not get_platform().exists(self.save_root):
            get_platform().makedirs(self.save_root)
        if self.log_file.exists():
            self.logger = MetricLogger.create_from_file(self.log_file, default_filename=False)
            self.load()
            self.update_last_save()
        # time operations
        self.stopwatch = Stopwatch(self.name())

    @property
    def save_root(self): return self.output_location / self.name()

    @property
    def log_file(self): return self.callback_file("runs.log")

    @property
    def log_key(self): return "callback_finished"

    @property
    def level_root(self) -> str:
        """The root of the main experiment on which this callback is based."""
        return self.desc.run_desc.run_path(self.replicate, self.level)

    @property
    def finished_steps(self):
        return [Step(i, self.iterations_per_epoch)
                 for i, _ in self.logger.get_data(self.log_key)]

    def _callback_function(self, output_location, step, *args, **kwds) -> None:
        """The method that is called to execute the callback.
        """
        assert self.output_location == Path(output_location), (self.output_location, output_location)
        # do not recompute metrics if already done
        if step not in self.finished_steps and self.schedule.do_run(step):
            if self.verbose: self.stopwatch.start()
            self.callback_function(output_location, step, *args, **kwds)
            self.logger.add(self.log_key, step, 1)
            if self.verbose: self.stopwatch.stop()
        # save metrics at same frequency as checkpoints, regardless of how many gen steps are run
        if self.last_save_ep < step.ep:
            if self.verbose: self.stopwatch.print()
            self.save_checkpoint()
            self.logger.save(self.log_file, default_filename=False)
            self.last_save_ep = step.ep

    def __call__(self, *args: Any, **kwds: Any) -> None:
        self._callback_function(*args, **kwds)

    def callback_file(self, key: str, step: Step=None, default_suffix=".npz"):
        key = Path(key)
        if len(key.suffix) == 0:  # no suffix, use default
            key = Path(key.name + default_suffix)
        if step is not None:  # insert step if set
            key = f"{key.stem}_{step.to_str()}{key.suffix}"
        return self.save_root / key

    # Interface that needs to be overridden for each callback.
    @abc.abstractmethod
    def load(self):
        raise NotImplementedError

    @abc.abstractmethod
    def save_checkpoint(self): raise NotImplementedError

    @abc.abstractmethod
    def callback_function(self, output_location, step, model, optimizer, logger, *args, **kwds):
        raise NotImplementedError

    @abc.abstractstaticmethod
    def name() -> str: raise NotImplementedError

    @abc.abstractstaticmethod
    def description() -> str: raise NotImplementedError
