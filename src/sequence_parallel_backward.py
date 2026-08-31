import torch
import torch.distributed as dist

from sp_ops import AllGatherSequence, ReduceScatterSequence


S, H = 4, 4


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if S % world or H % world:
        raise ValueError("Sequence and hidden sizes must divide evenly across ranks")

    torch.manual_seed(11)
    full_input = torch.arange(S * H, dtype=torch.float32).reshape(1, S, H).requires_grad_()
    full_weight = torch.randn(H, H, requires_grad=True)
    baseline = full_input @ full_weight.T
    baseline.sum().backward()

    local_input = full_input.detach().chunk(world, dim=1)[rank].clone().requires_grad_()
    local_weight = full_weight.detach().chunk(world, dim=1)[rank].clone().requires_grad_()
    gathered = AllGatherSequence.apply(local_input)
    hidden = gathered.chunk(world, dim=-1)[rank]
    partial = hidden @ local_weight.T
    local_output = ReduceScatterSequence.apply(partial)
    local_output.sum().backward()

    torch.testing.assert_close(local_output, baseline.detach().chunk(world, dim=1)[rank])
    torch.testing.assert_close(local_input.grad, full_input.grad.detach().chunk(world, dim=1)[rank])
    torch.testing.assert_close(local_weight.grad, full_weight.grad.detach().chunk(world, dim=1)[rank])
    print(f"Rank {rank}: SP forward and backward match baseline")
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
