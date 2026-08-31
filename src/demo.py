import os
import torch
import torch.distributed as dist

def main():
    # intialze the process group woth GLOO CPU backend communication (use NCCL for GPU backend)
    dist.init_process_group(backend='gloo')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank {rank} of {world_size} processes")

    # Give every rank a different local value.
    tensor = torch.tensor([float(rank + 1)])
    print(f"Rank {rank}: before all-reduce = {tensor.item()}")

    # Sum the values across all ranks and return that sum to every rank.
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank}: after all-reduce = {tensor.item()}")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
