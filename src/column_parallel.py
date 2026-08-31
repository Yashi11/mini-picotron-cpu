import torch
import torch.distributed as dist
from torch import nn


INPUT_FEATURES = 4
OUTPUT_FEATURES = 6


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if OUTPUT_FEATURES % world_size != 0:
        raise ValueError("OUTPUT_FEATURES must divide evenly across ranks")

    # Recreate the same deterministic weight matrix used by baseline.py.
    torch.manual_seed(7)
    baseline_layer = nn.Linear(INPUT_FEATURES, OUTPUT_FEATURES, bias=False)
    inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    # PyTorch stores Linear weights as [output_features, input_features].
    # Splitting rows here is equivalent to splitting output columns in X @ W notation.
    local_weight = baseline_layer.weight.chunk(world_size, dim=0)[rank]
    local_output = inputs @ local_weight.T

    # This all-gather is only for our correctness check. In a real TP MLP,
    # we would normally keep local_output sharded for the activation.
    output_shards = [torch.empty_like(local_output) for _ in range(world_size)]
    dist.all_gather(output_shards, local_output)
    distributed_output = torch.cat(output_shards, dim=-1)

    baseline_output = baseline_layer(inputs)
    torch.testing.assert_close(distributed_output, baseline_output)

    print(
        f"Rank {rank}: weight shard {tuple(local_weight.shape)}, "
        f"output shard {tuple(local_output.shape)}"
    )
    if rank == 0:
        print(f"Combined output matches baseline: {distributed_output}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
