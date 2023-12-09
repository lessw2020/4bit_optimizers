# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import numpy as np

import pytest
import torch
import torch.distributed as dist
from torch import nn, Tensor


def gpu_test(gpu_count: int = 1):
    """
    Annotation for GPU tests, skipping the test if the
    required amount of GPU is not available
    """
    message = f"Not enough GPUs to run the test: requires {gpu_count}"
    local_gpu_count: int = torch.cuda.device_count()
    return pytest.mark.skipif(local_gpu_count < gpu_count, reason=message)


def init_distributed_on_file(world_size: int, gpu_id: int, sync_file: str):
    """
    Init the process group need to do distributed training, by syncing
    the different workers on a file.
    """
    torch.cuda.set_device(gpu_id)
    dist.init_process_group(
        backend="nccl",
        init_method="file://" + sync_file,
        world_size=world_size,
        rank=gpu_id,
    )


@contextmanager
def with_temp_files(count: int):
    """
    Context manager to create temporary files and remove them
    after at the end of the context
    """
    if count == 1:
        fd, file_name = tempfile.mkstemp()
        yield file_name
        os.close(fd)
    else:
        temp_files = [tempfile.mkstemp() for _ in range(count)]
        yield [t[1] for t in temp_files]
        for t in temp_files:
            os.close(t[0])


def set_rng_seed(seed):
    """Sets the seed for pytorch and numpy random number generators"""
    torch.manual_seed(seed)
    random.seed(seed)


_ASSET_DIR = (Path(__file__).parent / "assets").resolve()


def get_asset_path(file_name: str) -> str:
    """Get the path to the file under assets directory."""
    return str(_ASSET_DIR.joinpath(file_name))


def assert_expected(
    actual: Any,
    expected: Any,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    check_device=True,
):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        check_device=check_device,
        msg=f"actual: {actual}, expected: {expected}",
    )


def tuple_to_dict(t: Tuple) -> Dict:
    if not isinstance(t, tuple):
        raise TypeError(f"Input must be of type tuple but got {type(t)}")

    return {k: v for k, v in enumerate(t)}