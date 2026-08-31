import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


FEATURES = 4
MICRO_BATCHES = 3


def make_linear_layers():
    torch.manual_seed(21)
    return (
        nn.Linear(FEATURES, FEATURES, bias=False),
        nn.Linear(FEATURES, FEATURES, bias=False),
    )


def full_batch():
    return torch.arange(1, MICRO_BATCHES * FEATURES + 1, dtype=torch.float32).reshape(
        MICRO_BATCHES, FEATURES
    )


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()

    if dist.get_world_size() != 2:
        raise ValueError("This AFAB example is written for exactly 2 ranks")

    # One unpipelined reference pass over the complete batch.
    reference_stage_0, reference_stage_1 = make_linear_layers()
    reference_inputs = full_batch().requires_grad_()
    reference_output = reference_stage_1(F.gelu(reference_stage_0(reference_inputs)))
    reference_output.sum().backward()

    stage_0, stage_1 = make_linear_layers()

    if rank == 0:
        inputs = full_batch().requires_grad_()
        hidden_activations = []

        # AFAB forward phase: send every micro-batch forward before any backward work.
        for micro_batch in range(MICRO_BATCHES):
            hidden = F.gelu(stage_0(inputs[micro_batch : micro_batch + 1]))
            hidden_activations.append(hidden)
            dist.send(hidden.detach(), dst=1)
            print(f"Rank 0: F{micro_batch} sent activation to rank 1")

        # AFAB backward phase: gradients return in reverse micro-batch order.
        for micro_batch in reversed(range(MICRO_BATCHES)):
            hidden_gradient = torch.empty_like(hidden_activations[micro_batch])
            dist.recv(hidden_gradient, src=1)
            hidden_activations[micro_batch].backward(hidden_gradient)
            print(f"Rank 0: B{micro_batch} received activation gradient")

        torch.testing.assert_close(stage_0.weight.grad, reference_stage_0.weight.grad)
        torch.testing.assert_close(inputs.grad, reference_inputs.grad)
        print("Rank 0: AFAB stage-0 and input gradients match baseline")

    else:
        received_hiddens = []
        outputs = []

        # AFAB forward phase: receive and process every micro-batch.
        for micro_batch in range(MICRO_BATCHES):
            hidden = torch.empty((1, FEATURES))
            dist.recv(hidden, src=0)
            hidden.requires_grad_()
            received_hiddens.append(hidden)
            outputs.append(stage_1(hidden))
            print(f"Rank 1: F{micro_batch} received activation from rank 0")

        # AFAB backward phase: finish all forwards, then backpropagate every micro-batch.
        for micro_batch in reversed(range(MICRO_BATCHES)):
            outputs[micro_batch].sum().backward()
            dist.send(received_hiddens[micro_batch].grad, dst=0)
            print(f"Rank 1: B{micro_batch} sent activation gradient to rank 0")

        torch.testing.assert_close(torch.cat(outputs), reference_output)
        torch.testing.assert_close(stage_1.weight.grad, reference_stage_1.weight.grad)
        print("Rank 1: AFAB stage-1 gradient matches baseline")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
