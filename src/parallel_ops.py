import torch
import torch.distributed as dist


class CopyToTensorParallelRegion(torch.autograd.Function):
    """Forward: identity. Backward: sum gradients from all TP ranks."""

    @staticmethod
    def forward(ctx, tensor):
        return tensor

    @staticmethod
    def backward(ctx, gradient):
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        return gradient


class ReduceFromTensorParallelRegion(torch.autograd.Function):
    """Forward: sum partial outputs. Backward: identity."""

    @staticmethod
    def forward(ctx, tensor):
        output = tensor.clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output

    @staticmethod
    def backward(ctx, gradient):
        return gradient
