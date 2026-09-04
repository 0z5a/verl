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

import argparse

import pytest

from scripts.benchmark_collective_phase_sweep import (
    _topology_has_nvlink,
    build_group_specs,
    parse_size,
    percentile,
    resolve_group_layout,
)


def test_parse_size():
    assert parse_size("128MiB") == 128 * 1024**2
    assert parse_size("1.5KB") == 1500
    with pytest.raises(argparse.ArgumentTypeError):
        parse_size("0")


def test_general_mesh_layout():
    groups_a, groups_b = build_group_specs("mesh", 6, (2, 3))
    assert [group.ranks for group in groups_a] == [(0, 1, 2), (3, 4, 5)]
    assert [group.ranks for group in groups_b] == [(0, 3), (1, 4), (2, 5)]
    assert all(any(rank in group.ranks for group in groups_a) for rank in range(6))
    assert all(any(rank in group.ranks for group in groups_b) for rank in range(6))


@pytest.mark.parametrize(
    ("world_size", "expected"),
    [(4, "mesh-2x2"), (6, "mesh-2x3"), (8, "mesh-2x4"), (12, "mesh-3x4")],
)
def test_auto_layout_is_not_tied_to_six_ranks(world_size, expected):
    assert resolve_group_layout("auto", world_size) == (
        expected,
        tuple(int(value) for value in expected.removeprefix("mesh-").split("x")),
    )


def test_prime_world_size_auto_falls_back_to_overlapping_world_groups():
    groups_a, groups_b = build_group_specs("auto", 7)
    assert groups_a == [groups_a[0]]
    assert groups_b == [groups_b[0]]
    assert groups_a[0].ranks == groups_b[0].ranks == tuple(range(7))


def test_ep_dp_shorthand_is_general():
    assert build_group_specs("ep4-dp2", 8) == build_group_specs("mesh", 8, (2, 4))


def test_mesh_shape_must_match_world_size():
    with pytest.raises(ValueError, match="does not match world size"):
        build_group_specs("mesh", 6, (2, 4))


def test_percentile_uses_linear_interpolation():
    assert percentile([], 50) is None
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == pytest.approx(3.85)


def test_topology_detection_ignores_nvlink_legend():
    pcie_topology = "GPU0 GPU1\nGPU0 X PIX\nGPU1 PIX X\nLegend:\nNV# = Connection traversing bonded NVLinks"
    nvlink_topology = "GPU0 GPU1\nGPU0 X NV4\nGPU1 NV4 X\nLegend:\nNV# = Connection traversing bonded NVLinks"
    assert not _topology_has_nvlink(pcie_topology)
    assert _topology_has_nvlink(nvlink_topology)
