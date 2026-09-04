# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Low-overhead semantic NVTX ranges for distributed communication.

Set ``VERL_COMM_TRACE=0`` before importing this module to disable the ranges.
The default is enabled so an Nsight Systems run contains semantic collective
names without requiring a second, differently configured training run.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

import torch
import torch.distributed as dist

_TRACE_FIELDS = (
    "step",
    "microbatch",
    "layer",
    "direction",
    "message_bytes",
    "process_group_id",
    "logical_sequence_id",
    "requested_offset_us",
)
_COMM_TRACE_ENABLED = os.environ.get("VERL_COMM_TRACE", "1").lower() not in {"0", "false", "off", "no"}
_COMM_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar("verl_comm_trace_context", default=None)
_SEQUENCE_LOCK = threading.Lock()
_SEQUENCE_IDS: dict[tuple[int, str | None], int] = {}


def _escape(value: Any) -> str:
    return str(value).replace("%", "%25").replace("|", "%7C").replace("=", "%3D")


def _process_group_id(group: dist.ProcessGroup | None) -> str:
    if group is None:
        if not dist.is_initialized():
            return "world"
        group = dist.group.WORLD
    name = getattr(group, "group_name", None)
    if name is not None:
        return str(name)
    return "world" if group is dist.group.WORLD else f"group-size-{dist.get_world_size(group)}"


def _next_sequence_id(group: dist.ProcessGroup | None, process_group_id: str | None = None) -> int:
    if group is None and dist.is_initialized():
        group = dist.group.WORLD
    key = (id(group), process_group_id)
    with _SEQUENCE_LOCK:
        sequence_id = _SEQUENCE_IDS.get(key, 0)
        _SEQUENCE_IDS[key] = sequence_id + 1
    return sequence_id


def _nvtx_available() -> bool:
    """Return whether this process can emit NVIDIA NVTX ranges."""

    return (
        _COMM_TRACE_ENABLED
        and getattr(torch.version, "hip", None) is None
        and torch.cuda.is_available()
        and callable(getattr(getattr(torch.cuda, "nvtx", None), "range", None))
    )


def format_communication_range(operation: str, **metadata: Any) -> str:
    """Build a deterministic, parseable ``verl.comm/*`` range label."""

    if not operation or "|" in operation or "=" in operation:
        raise ValueError("operation must be non-empty and cannot contain '|' or '='")
    unknown = metadata.keys() - _TRACE_FIELDS
    if unknown:
        raise ValueError(f"unsupported communication trace fields: {sorted(unknown)}")
    fields = [f"{name}={_escape(metadata[name])}" for name in _TRACE_FIELDS if metadata.get(name) is not None]
    return "|".join((f"verl.comm/{operation}", *fields))


@contextmanager
def communication_trace_context(**metadata: Any) -> Iterator[None]:
    """Attach step/microbatch/layer metadata to nested communication ranges."""

    unknown = metadata.keys() - _TRACE_FIELDS
    if unknown:
        raise ValueError(f"unsupported communication trace fields: {sorted(unknown)}")
    merged = {**(_COMM_CONTEXT.get() or {}), **{key: value for key, value in metadata.items() if value is not None}}
    token = _COMM_CONTEXT.set(merged)
    try:
        yield
    finally:
        _COMM_CONTEXT.reset(token)


@contextmanager
def communication_nvtx_range(
    operation: str,
    *,
    tensor: torch.Tensor | None = None,
    group: dist.ProcessGroup | None = None,
    step: int | None = None,
    microbatch: int | None = None,
    layer: int | str | None = None,
    direction: str | None = None,
    message_bytes: int | None = None,
    process_group_id: str | None = None,
    logical_sequence_id: int | str | None = None,
    requested_offset_us: int | float | None = None,
) -> Iterator[None]:
    """Open a semantic NVTX range around a collective launch.

    The range describes the framework operation rather than the transport
    kernel. Nsight Systems can therefore associate the enclosed NCCL API call
    with its Ulysses/FSDP/Megatron meaning.
    """

    if not _nvtx_available():
        yield
        return
    metadata = dict(_COMM_CONTEXT.get() or {})
    resolved_group_id = process_group_id if process_group_id is not None else _process_group_id(group)
    metadata.update(
        {
            "step": step if step is not None else metadata.get("step"),
            "microbatch": microbatch if microbatch is not None else metadata.get("microbatch"),
            "layer": layer if layer is not None else metadata.get("layer"),
            "direction": direction if direction is not None else metadata.get("direction"),
            "message_bytes": (
                message_bytes
                if message_bytes is not None
                else tensor.numel() * tensor.element_size()
                if tensor is not None
                else metadata.get("message_bytes")
            ),
            "process_group_id": resolved_group_id,
            "logical_sequence_id": (
                logical_sequence_id
                if logical_sequence_id is not None
                else metadata["logical_sequence_id"]
                if metadata.get("logical_sequence_id") is not None
                else _next_sequence_id(group, resolved_group_id)
            ),
            "requested_offset_us": (
                requested_offset_us if requested_offset_us is not None else metadata.get("requested_offset_us")
            ),
        }
    )
    message = format_communication_range(operation, **metadata)
    with torch.cuda.nvtx.range(message):
        yield
