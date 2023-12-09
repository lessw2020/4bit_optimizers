# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from tests.test_utils import assert_expected, gpu_test, set_rng_seed
from torchmultimodal.modules.optimizers.anyprecision import AnyPrecisionAdamW


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(2020)
