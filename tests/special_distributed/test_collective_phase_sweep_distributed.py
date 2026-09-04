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
"""Two-rank CPU smoke test for the collective phase-sweep benchmark."""

import json
import subprocess
import sys
from pathlib import Path


def test_collective_phase_sweep_two_rank_smoke(tmp_path: Path):
    repository_root = Path(__file__).parents[2]
    output = tmp_path / "phase_sweep.json"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        str(repository_root / "scripts" / "benchmark_collective_phase_sweep.py"),
        "--backend",
        "gloo",
        "--device",
        "cpu",
        "--group-layout",
        "world",
        "--comm-a",
        "all-to-all",
        "--comm-b",
        "all-reduce",
        "--message-bytes-a",
        "4KiB",
        "--message-bytes-b",
        "4KiB",
        "--policies",
        "isolated",
        "concurrent",
        "serialized",
        "offset",
        "--offset-us",
        "-1000",
        "0",
        "1000",
        "--warmup",
        "1",
        "--iters",
        "2",
        "--no-shuffle-offsets",
        "--output-json",
        str(output),
    ]
    subprocess.run(command, cwd=repository_root, check=True, timeout=120)

    payload = json.loads(output.read_text())
    assert payload["schema_version"] == 2
    assert payload["world_size"] == 2
    assert payload["timestamp_domain"] == "single-node-perf-counter"
    assert payload["gpu_timestamp_semantics"] == "event-bracket"
    assert payload["kernel_observed"] is False
    assert payload["timing_sources"]["realized_gpu_offset"] == "host_call_bracket_start"
    assert payload["sequence_validation"]["all_groups_consistent"]
    assert {result["policy"] for result in payload["results"]} == {
        "isolated",
        "concurrent",
        "serialized",
        "offset",
    }
    assert all(result.keys() == payload["results"][0].keys() for result in payload["results"])
    offset_results = [result for result in payload["results"] if result["policy"] == "offset"]
    assert [result["requested_offset_us"] for result in offset_results] == [-1000.0, 0.0, 1000.0]
    assert offset_results[0]["realized_gpu_offset_us_p50"] < 0
    assert offset_results[-1]["realized_gpu_offset_us_p50"] > 0
    required_metrics = {
        "api_launch_offset_us_p50",
        "realized_gpu_offset_us_p50",
        "rank_start_skew_us_p95",
        "rank_finish_skew_us_p95",
        "launch_anchor_lateness_us_p95",
        "actual_overlap_ms",
        "pair_completion_ms",
        "stretch_a",
        "stretch_b",
    }
    assert required_metrics <= offset_results[0].keys()
