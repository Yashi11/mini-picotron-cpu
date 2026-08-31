import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_tensor_parallel_forward_and_backward_match_baseline():
    environment = os.environ.copy()
    environment["GLOO_SOCKET_IFNAME"] = "lo0"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc_per_node=2",
        "--node_rank=0",
        "--rdzv_backend=static",
        "--master_addr=127.0.0.1",
        "--master_port=29502",
        str(PROJECT_ROOT / "src" / "tensor_parallel_backward.py"),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert "Rank 0: output and all local gradients match the baseline" in completed.stdout
    assert "Rank 1: output and all local gradients match the baseline" in completed.stdout
