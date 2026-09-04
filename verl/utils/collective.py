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
"""Common primitives for asynchronous distributed collectives."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, cast

import torch.distributed as dist

T = TypeVar("T")
_UNSET = object()
_SEQUENCE_LOCK = threading.Lock()
_SEQUENCE_IDS: dict[tuple[int, str], int] = {}


class CollectiveWork(Protocol):
    """Minimum interface implemented by ``torch.distributed.Work``."""

    def wait(self) -> object: ...


def resolve_process_group_id(group: dist.ProcessGroup | None = None) -> str:
    """Return a useful process-group identifier without requiring private APIs."""

    if group is None:
        if not dist.is_initialized():
            return "world"
        group = dist.group.WORLD
    name = getattr(group, "group_name", None)
    if name is not None:
        return str(name)
    return "world" if group is dist.group.WORLD else f"group-size-{dist.get_world_size(group)}"


def next_collective_sequence_id(group: dist.ProcessGroup | None = None, process_group_id: str | None = None) -> int:
    """Allocate a rank-local sequence number for one communicator.

    Ranks that launch collectives in the same order obtain matching IDs. A
    trace or benchmark can use the IDs to detect divergent launch order.
    """

    if group is None and dist.is_initialized():
        group = dist.group.WORLD
    resolved_group_id = process_group_id if process_group_id is not None else resolve_process_group_id(group)
    key = (id(group), resolved_group_id)
    with _SEQUENCE_LOCK:
        sequence_id = _SEQUENCE_IDS.get(key, 0)
        _SEQUENCE_IDS[key] = sequence_id + 1
    return sequence_id


@dataclass(slots=True)
class AsyncCollectiveHandle(Generic[T]):
    """Own an async collective and its post-communication transformation.

    ``wait_collective`` waits for transport completion and records
    ``complete_event`` before any concat/reshape finalizer runs. Callers that
    need separate measurements can invoke ``wait_collective`` and
    ``finalize_result`` independently; ``wait`` provides the usual combined
    behavior. All three methods are idempotent.
    """

    work: CollectiveWork
    finalize: Callable[[], T]
    comm_kind: str
    process_group_id: str
    sequence_id: int
    launch_event: Any | None = None
    complete_event: Any | None = None
    _collective_complete: bool = field(default=False, init=False, repr=False)
    _result: object = field(default=_UNSET, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def collective_complete(self) -> bool:
        return self._collective_complete

    @property
    def finalized(self) -> bool:
        return self._result is not _UNSET

    def wait_collective(self) -> None:
        """Wait once for the collective and mark its completion event."""

        with self._lock:
            if self._collective_complete:
                return
            self.work.wait()
            if self.complete_event is not None:
                record = getattr(self.complete_event, "record", None)
                if not callable(record):
                    raise TypeError("complete_event must provide a callable record() method")
                record()
            self._collective_complete = True

    def finalize_result(self) -> T:
        """Wait for communication if needed, then run the finalizer once."""

        self.wait_collective()
        with self._lock:
            if self._result is _UNSET:
                self._result = self.finalize()
            return cast(T, self._result)

    def wait(self) -> T:
        """Wait for communication and return the finalized result."""

        return self.finalize_result()
