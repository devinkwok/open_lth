# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path

from foundations import paths
from foundations.step import Step
from platforms.platform import get_platform


class MetricLogger:
    def __init__(self):
        self.log = {}

    def add(self, name: str, step: Step, value: float):
        self.log[(name, step.iteration)] = value

    def __str__(self):
        return '\n'.join(['{},{},{}'.format(k[0], k[1], v) for k, v in self.log.items()])

    @staticmethod
    def create_from_string(as_str):
        logger = MetricLogger()
        if len(as_str.strip()) == 0:
            return logger

        rows = [row.split(',') for row in as_str.strip().split('\n')]
        logger.log = {(name, int(iteration)): float(value) for name, iteration, value in rows}
        return logger

    @staticmethod
    def create_from_file(filename, default_filename=True):
        file = paths.logger(filename) if default_filename else filename
        with get_platform().open(file) as fp:
            as_str = fp.read()
        return MetricLogger.create_from_string(as_str)

    def save(self, location, default_filename=True):
        location = Path(location)
        file = paths.logger(location) if default_filename else location
        if not get_platform().is_primary_process: return
        if not get_platform().exists(file.parent):
            get_platform().makedirs(file.parent)
        with get_platform().open(file, 'w') as fp:
            fp.write(str(self))

    def get_data(self, desired_name):
        d = {k[1]: v for k, v in self.log.items() if k[0] == desired_name}
        return [(k, d[k]) for k in sorted(d.keys())]
