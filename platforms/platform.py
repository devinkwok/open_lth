# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import platforms.local

_PLATFORM = platforms.local.Platform()


def get_platform():
    return _PLATFORM
