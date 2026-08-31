"""Command-line learning journey for the Mini Picotron CPU lab."""

import argparse
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Lesson:
    title: str
    script: str
    ranks: int
    goal: str
    look_for: str


LESSONS = {
    "collectives": Lesson(
        "1. All-reduce foundations",
        "demo.py",
        2,
        "See independent ranks combine local tensors into one shared result.",
        "Both ranks print 3.0 after all-reduce.",
    ),
    "column-parallel": Lesson(
        "2. Column-parallel linear",
        "column_parallel.py",
        2,
        "Split output features across ranks and rebuild the baseline output.",
        "Two (3, 4) weight shards and a combined output matching the baseline.",
    ),
    "tensor-parallel-mlp": Lesson(
        "3. Tensor-parallel MLP",
        "tensor_parallel_mlp.py",
        2,
        "Keep hidden features sharded, then all-reduce the row-parallel output.",
        "Each rank has a partial (1, 4) output; all-reduce matches baseline.",
    ),
    "tensor-parallel-backward": Lesson(
        "4. Tensor-parallel backward pass",
        "tensor_parallel_backward.py",
        2,
        "Verify forward values and every relevant gradient against the baseline.",
        "Both ranks report that local gradients match the baseline.",
    ),
    "sequence-parallel": Lesson(
        "5. Sequence-parallel layout",
        "sequence_parallel_layout.py",
        2,
        "All-gather token shards for TP work, then reduce-scatter them back.",
        "(1, 2, 4) → (1, 4, 4) → (1, 2, 4).",
    ),
    "pipeline-forward": Lesson(
        "6. Pipeline-parallel forward pass",
        "pipeline_parallel.py",
        2,
        "Send an activation from the first model stage to the second stage.",
        "Rank 0 sends an activation and the pipeline output matches baseline.",
    ),
    "pipeline-backward": Lesson(
        "7. Pipeline-parallel backward pass",
        "pipeline_backward.py",
        2,
        "Send the activation gradient upstream and verify both stages' gradients.",
        "Rank 1 sends an activation gradient; rank 0 matches baseline gradients.",
    ),
    "pipeline-afab": Lesson(
        "8. AFAB micro-batch schedule",
        "pipeline_afab.py",
        2,
        "Run all micro-batch forwards, then all backwards in reverse order.",
        "F0, F1, F2 followed by B2, B1, B0 and matching gradients.",
    ),
    "1f1b": Lesson("9. One-forward, one-backward", "schedule_demo.py", 1, "Alternate forward and backward work after pipeline warmup.", "Each stage reports warmup, steady-state, and drain events."),
    "interleaved": Lesson("10. Interleaved pipeline", "schedule_demo.py", 1, "Map virtual model chunks back onto physical pipeline stages.", "Each stage revisits virtual chunks 0 and 1."),
}


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run_lesson(name: str) -> None:
    lesson = LESSONS[name]
    environment = os.environ.copy()
    if platform.system() == "Darwin":
        environment["GLOO_SOCKET_IFNAME"] = "lo0"

    port = free_local_port()
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc_per_node={lesson.ranks}",
        "--node_rank=0",
        "--rdzv_backend=static",
        "--master_addr=127.0.0.1",
        f"--master_port={port}",
        str(PROJECT_ROOT / "src" / lesson.script),
    ]
    if name == "1f1b": command.append("--mode=1f1b")
    if name == "interleaved": command.append("--mode=interleaved")

    print(f"\n{lesson.title}\n")
    print(f"Goal: {lesson.goal}")
    print(f"Look for: {lesson.look_for}\n")
    subprocess.run(command, check=True, env=environment)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a guided Mini Picotron CPU lesson.")
    parser.add_argument("lesson", choices=["list", *LESSONS], help="Lesson to run, or 'list'.")
    args = parser.parse_args()

    if args.lesson == "list":
        print("Mini Picotron CPU lessons:\n")
        for name, lesson in LESSONS.items():
            print(f"{name:26} {lesson.title}")
        return

    run_lesson(args.lesson)


if __name__ == "__main__":
    main()
