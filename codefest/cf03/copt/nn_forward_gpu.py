"""
nn_forward_gpu.py

Forward pass of a small feedforward network on the GPU.

Architecture (per Figure 3):
    input  [batch, 4]
      -> Linear(4, 5) -> ReLU
      -> Linear(5, 1)  (no activation)
    output [batch, 1]

Batch size: 16. No training; this script just builds the net, generates a
random input batch, moves everything to the GPU, runs one forward pass, and
verifies the output shape and device.

Course: ECE 410/510, Spring 2026 — codefest cf03 / COPT
"""

import sys
import torch
import torch.nn as nn


def main() -> int:
    # ---- 1. device detection ----------------------------------------------
    # Use the exact form requested by the spec.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        # No GPU available: print a clear message and exit without running
        # the (non-existent) training loop, per the spec.
        print("No CUDA GPU detected. This script requires a GPU. Exiting.")
        return 1

    # Print the GPU's name. torch.cuda.get_device_name takes either an int
    # index or a cuda device; device.index may be None when unspecified, so
    # fall back to device 0.
    gpu_index = device.index if device.index is not None else 0
    print(f"Using device: {device} ({torch.cuda.get_device_name(gpu_index)})")

    # ---- 2. define the network --------------------------------------------
    # 4 -> 5 (ReLU) -> 1 (linear). Sequential is the shortest expression.
    model = nn.Sequential(
        nn.Linear(in_features=4, out_features=5),
        nn.ReLU(),
        nn.Linear(in_features=5, out_features=1),
    ).to(device)

    # Quick sanity print so the terminal output documents the exact model.
    print("Model:")
    print(model)

    # ---- 3. random input batch on the GPU ---------------------------------
    # Shape [16, 4]: 16 input vectors, each of length 4.
    # Seed so the output is reproducible run-to-run.
    torch.manual_seed(0)
    x = torch.randn(16, 4).to(device)
    print(f"Input shape:  {tuple(x.shape)}   device: {x.device}")

    # ---- 4. forward pass --------------------------------------------------
    # torch.no_grad() because we're not training; skips building the autograd
    # graph and saves a bit of time/memory. Optional but tidy.
    with torch.no_grad():
        y = model(x)

    # ---- 5. verify output shape and device --------------------------------
    print(f"Output shape: {tuple(y.shape)}   device: {y.device}")

    # Hard check against the spec: output must be [16, 1] and on CUDA.
    assert tuple(y.shape) == (16, 1), f"expected output shape (16, 1), got {tuple(y.shape)}"
    assert y.device.type == "cuda",  f"expected output on cuda, got {y.device}"
    print("Shape and device checks passed.")

    # Print the 16 output values for the record.
    print("Output values:")
    # .cpu() so printing doesn't implicitly sync / format a CUDA tensor
    # surprisingly; squeeze the trailing dim for a clean 1-D print.
    print(y.squeeze(-1).cpu().numpy())

    return 0


if __name__ == "__main__":
    sys.exit(main())
