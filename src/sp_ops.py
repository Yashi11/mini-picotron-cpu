import torch
import torch.distributed as dist


class AllGatherSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local):
        ctx.world = dist.get_world_size()
        parts = [torch.empty_like(local) for _ in range(ctx.world)]
        dist.all_gather(parts, local)
        return torch.cat(parts, dim=1)

    @staticmethod
    def backward(ctx, gradient):
        parts = list(gradient.chunk(ctx.world, dim=1))
        local = torch.empty_like(parts[0])
        dist.reduce_scatter(local, parts, op=dist.ReduceOp.SUM)
        return local


class ReduceScatterSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, partial):
        ctx.world = dist.get_world_size()
        parts = list(partial.chunk(ctx.world, dim=1))
        local = torch.empty_like(parts[0])
        dist.reduce_scatter(local, parts, op=dist.ReduceOp.SUM)
        return local

    @staticmethod
    def backward(ctx, gradient):
        parts = [torch.empty_like(gradient) for _ in range(ctx.world)]
        dist.all_gather(parts, gradient)
        return torch.cat(parts, dim=1)
