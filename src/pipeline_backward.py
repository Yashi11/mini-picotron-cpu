import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


FEATURES = 4


def make_linear_layers():
    """Create two deterministic layers with identical weights on every rank."""
    torch.manual_seed(21)
    return (
        nn.Linear(FEATURES, FEATURES, bias=False),
        nn.Linear(FEATURES, FEATURES, bias=False),
    )


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()

    if dist.get_world_size() != 2:
        raise ValueError("This first pipeline example is written for exactly 2 ranks")

    # Reference model: used only to calculate the answer we must reproduce.
    reference_stage_0, reference_stage_1 = make_linear_layers()
    reference_inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    reference_output = reference_stage_1(F.gelu(reference_stage_0(reference_inputs)))
    reference_output.sum().backward()

    # Pipeline model: rank 0 uses stage_0; rank 1 uses stage_1.
    stage_0, stage_1 = make_linear_layers()

    if rank == 0:
        inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        hidden = F.gelu(stage_0(inputs))

        # Forward direction: send an activation without its local autograd graph.
        dist.send(hidden.detach(), dst=1)

        # Backward direction: receive d(loss)/d(hidden) from stage 1.
        hidden_gradient = torch.empty_like(hidden)
        dist.recv(hidden_gradient, src=1)
        hidden.backward(hidden_gradient)

        torch.testing.assert_close(stage_0.weight.grad, reference_stage_0.weight.grad)
        torch.testing.assert_close(inputs.grad, reference_inputs.grad)
        print("Rank 0: stage-0 and input gradients match the baseline")

    else:
        # Receiving creates a new tensor, so this rank starts a new local autograd graph.
        received_hidden = torch.empty((1, FEATURES))
        dist.recv(received_hidden, src=0)
        received_hidden.requires_grad_()
        pipeline_output = stage_1(received_hidden)
        pipeline_output.sum().backward()

        torch.testing.assert_close(pipeline_output, reference_output)
        torch.testing.assert_close(stage_1.weight.grad, reference_stage_1.weight.grad)

        # Send d(loss)/d(hidden) upstream so rank 0 can continue backward.
        dist.send(received_hidden.grad, dst=0)
        print("Rank 1: stage-1 gradient matches baseline; sent activation gradient to rank 0")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
