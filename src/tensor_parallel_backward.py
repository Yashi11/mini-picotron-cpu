import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from parallel_ops import CopyToTensorParallelRegion, ReduceFromTensorParallelRegion


INPUT_FEATURES = 4
HIDDEN_FEATURES = 6
OUTPUT_FEATURES = 4


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if HIDDEN_FEATURES % world_size != 0:
        raise ValueError("HIDDEN_FEATURES must divide evenly across ranks")

    torch.manual_seed(7)
    first_linear = nn.Linear(INPUT_FEATURES, HIDDEN_FEATURES, bias=False)
    second_linear = nn.Linear(HIDDEN_FEATURES, OUTPUT_FEATURES, bias=False)

    # The unsharded reference forward and backward pass.
    baseline_inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    baseline_output = second_linear(F.gelu(first_linear(baseline_inputs)))
    baseline_output.sum().backward()

    # Each rank keeps only the shards it would own in a real TP MLP.
    first_weight_shard = first_linear.weight.detach().chunk(world_size, dim=0)[rank]
    first_weight_shard = first_weight_shard.clone().requires_grad_()
    second_weight_shard = second_linear.weight.detach().chunk(world_size, dim=1)[rank]
    second_weight_shard = second_weight_shard.clone().requires_grad_()
    inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)

    # Column-parallel first linear; its input is replicated across TP ranks.
    local_hidden = F.gelu(CopyToTensorParallelRegion.apply(inputs) @ first_weight_shard.T)

    # Row-parallel second linear; every rank contributes to every output feature.
    local_output = local_hidden @ second_weight_shard.T
    distributed_output = ReduceFromTensorParallelRegion.apply(local_output)
    distributed_output.sum().backward()

    # Compare each local shard gradient to the matching slice of the baseline.
    expected_first_grad = first_linear.weight.grad.chunk(world_size, dim=0)[rank]
    expected_second_grad = second_linear.weight.grad.chunk(world_size, dim=1)[rank]
    torch.testing.assert_close(distributed_output, baseline_output)
    torch.testing.assert_close(first_weight_shard.grad, expected_first_grad)
    torch.testing.assert_close(second_weight_shard.grad, expected_second_grad)
    torch.testing.assert_close(inputs.grad, baseline_inputs.grad)

    print(f"Rank {rank}: output and all local gradients match the baseline")
    if rank == 0:
        print(f"Input gradient matches baseline: {inputs.grad}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
