# Mini Picotron CPU Lab

An executable learning lab for distributed Transformer training concepts. It runs
on one CPU-only computer, but uses real PyTorch processes and collectives.

## What this is

Each lesson has four parts:

1. A small distributed implementation.
2. Rank-by-rank output that makes ownership and communication visible.
3. An unsharded PyTorch baseline.
4. Assertions that prove the distributed output and gradients are correct.

This teaches distributed *semantics* and correctness. It does not measure CUDA,
NCCL, NVLink, or multi-node GPU performance.

## Start here

Activate the local environment, then list the guided lessons:

```bash
source cpuvenv/bin/activate
python src/lab.py list
```

Run one lesson:

```bash
python src/lab.py tensor-parallel-mlp
```

The lab picks a local rendezvous port and launches the required CPU ranks for
you. You no longer need to type the full `torchrun` command for every lesson.

## Visual lesson interface

The companion interface turns the same lessons into a rank-by-rank stepper.

```bash
cd learning-ui
npm run dev
```

Open the local address shown in the terminal. Use the stepper to inspect the
layout and communication first, then run the exact matching CPU experiment
from the displayed command.

## Learning path

| Lesson | You learn |
| --- | --- |
| `collectives` | Ranks, world size, and all-reduce |
| `column-parallel` | Splitting a linear layer by output features |
| `tensor-parallel-mlp` | Column-parallel → GELU → row-parallel → all-reduce |
| `tensor-parallel-backward` | Why forward and backward communication are conjugate pairs |
| `sequence-parallel` | All-gather into TP and reduce-scatter back to SP |
| `pipeline-forward` | Splitting model depth and sending activations |
| `pipeline-backward` | Returning activation gradients upstream |
| `pipeline-afab` | All-forward, all-backward micro-batch scheduling |

## Verification

The `tests/` directory contains automated correctness tests. Run them with:

```bash
pytest -q
```
