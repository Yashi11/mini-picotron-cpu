import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_all_reduce_sums_values_on_every_rank():
    """Two ranks start with 1 and 2, then both must receive their sum: 3."""
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
        "--master_port=29501",
        str(PROJECT_ROOT / "src" / "demo.py"),
    ]

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert "Rank 0: after all-reduce = 3.0" in completed.stdout
    assert "Rank 1: after all-reduce = 3.0" in completed.stdout
