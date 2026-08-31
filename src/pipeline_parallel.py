import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


FEATURES = 4


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size != 2:
        raise ValueError("This first pipeline example is written for exactly 2 ranks")

    # Two sequential model stages. Rank 0 uses stage 0; rank 1 uses stage 1.
    torch.manual_seed(21)
    stage_0 = nn.Linear(FEATURES, FEATURES, bias=False)
    stage_1 = nn.Linear(FEATURES, FEATURES, bias=False)

    if rank == 0:
        inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        hidden = F.gelu(stage_0(inputs))

        # Hand the activation to the next pipeline stage.
        dist.send(hidden, dst=1)

        pipeline_output = torch.empty_like(hidden)
        dist.recv(pipeline_output, src=1)

        baseline_output = stage_1(F.gelu(stage_0(inputs)))
        torch.testing.assert_close(pipeline_output, baseline_output)
        print(f"Rank 0: stage 0 sent activation {tuple(hidden.shape)} to rank 1")
        print(f"Pipeline output matches baseline: {pipeline_output}")

    else:
        hidden = torch.empty((1, FEATURES))
        dist.recv(hidden, src=0)
        pipeline_output = stage_1(hidden)

        # Return the final output only so rank 0 can verify against its baseline.
        dist.send(pipeline_output, dst=0)
        print(f"Rank 1: stage 1 received activation {tuple(hidden.shape)} from rank 0")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
