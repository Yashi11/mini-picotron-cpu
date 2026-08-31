import torch
from torch import nn


def main():
    torch.manual_seed(7)

    # One ordinary, unsharded linear layer: 4 input features -> 6 output features.
    layer = nn.Linear(in_features=4, out_features=6, bias=False)
    inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)

    outputs = layer(inputs)
    loss = outputs.sum()
    loss.backward()

    print(f"input shape: {tuple(inputs.shape)}")
    print(f"weight shape: {tuple(layer.weight.shape)}")
    print(f"output shape: {tuple(outputs.shape)}")
    print(f"output: {outputs.detach()}")
    print(f"weight-gradient shape: {tuple(layer.weight.grad.shape)}")
    print(f"input gradient: {inputs.grad}")


if __name__ == "__main__":
    main()
