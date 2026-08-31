import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


INPUT_FEATURES = 4
HIDDEN_FEATURES = 6
OUTPUT_FEATURES = 4


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if HIDDEN_FEATURES % world_size != 0:
        raise ValueError("HIDDEN_FEATURES must divide evenly across ranks")

    # The two ordinary layers are our deterministic reference MLP.
    torch.manual_seed(7)
    first_linear = nn.Linear(INPUT_FEATURES, HIDDEN_FEATURES, bias=False)
    second_linear = nn.Linear(HIDDEN_FEATURES, OUTPUT_FEATURES, bias=False)
    inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    # 1) Column-parallel first linear: each rank owns hidden-feature outputs.
    first_weight_shard = first_linear.weight.chunk(world_size, dim=0)[rank]
    local_hidden = inputs @ first_weight_shard.T
    local_hidden = F.gelu(local_hidden)

    # 2) Row-parallel second linear: each rank owns matching hidden-feature inputs.
    second_weight_shard = second_linear.weight.chunk(world_size, dim=1)[rank]
    local_output = local_hidden @ second_weight_shard.T

    # Every rank has a partial contribution to all output features. Sum them.
    distributed_output = local_output.clone()
    dist.all_reduce(distributed_output, op=dist.ReduceOp.SUM)

    # Verify against the ordinary, unsharded MLP.
    baseline_output = second_linear(F.gelu(first_linear(inputs)))
    torch.testing.assert_close(distributed_output, baseline_output)

    print(
        f"Rank {rank}: first-weight {tuple(first_weight_shard.shape)}, "
        f"local hidden {tuple(local_hidden.shape)}, "
        f"second-weight {tuple(second_weight_shard.shape)}, "
        f"partial output {tuple(local_output.shape)}"
    )
    if rank == 0:
        print(f"All-reduced output matches baseline: {distributed_output}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
