import abc
from time import time
from typing import Any
from pathlib import Path

from foundations.step import Step
from training.metric_logger import MetricLogger


class ExecTime:
    def __init__(self, message, verbose: bool = True) -> None:
        self.message = message
        self.verbose = verbose

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.verbose:
            print(f"{self.message} {(time() - self.start):0.4f} seconds")


class CallbackSchedule:

    def __init__(self, steps=None) -> None:
        self.steps = steps

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
                step_list.append(Step.from_str(element))
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
        if self.log_file.exists():
            self.logger = MetricLogger.create_from_file(self.log_file)
            self.load(self.output_location)
            self.update_last_save()

    @property
    def log_file(self): return self.output_location / self.name()

    @property
    def log_name(self): return "callback_finished"

    @property
    def level_root(self) -> str:
        """The root of the main experiment on which this callback is based."""
        return self.desc.run_desc.run_path(self.replicate, self.level)

    @property
    def finished_steps(self):
        return [Step(i, self.iterations_per_epoch)
                 for i, _ in self.logger.get_data(self.log_name)]

    def update_last_save(self):
        self.last_save = max(self.logger).ep
        self.logger.save(self.log_file)

    def _callback_function(self, output_location, step, *args, **kwds) -> None:
        """The method that is called to execute the callback.
        """
        assert self.output_location == output_location
        # do not recompute metrics if already done
        if step not in self.finished_steps and self.schedule.do_run(step):
            self.generate_metrics(step)
            self.logger.add(self.log_name, step, 1)
        # save metrics at same frequency as checkpoints, regardless of how many gen steps are run
        if self.last_save_ep < step.ep:
            self._save_metrics(output_location)
            self.update_last_save()

    def __call__(self, *args: Any, **kwds: Any) -> None:
        self._callback_function(*args, **kwds)

    def callback_file(self, key: str): return f"{self.name()}_{key}"

    # Interface that needs to be overridden for each callback.
    @abc.abstractmethod
    def load(self, output_location):
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, output_location): raise NotImplementedError

    @abc.abstractmethod
    def callback_function(self, output_location, step, model, optimizer, logger, *args, **kwds):
        raise NotImplementedError

    @abc.abstractmethod
    def name(self) -> str: raise NotImplementedError

    @abc.abstractstaticmethod
    def description() -> str: raise NotImplementedError
