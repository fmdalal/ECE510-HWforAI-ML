import torch
import torchvision.models as models
from torchinfo import summary
import os

# Setup
os.makedirs("codefest/cf01/profiling", exist_ok=True)
model = models.resnet18(weights=None)   # or weights="IMAGENET1K_V1" for pretrained
model.eval()

# Run profiling
stats = summary(
    model,
    input_size=(1, 3, 224, 224),
    col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds", "trainable"],
    col_width=20,
    depth=5,
    row_settings=["var_names"],
    verbose=0,     # suppress console output; we'll write to file
    device="cpu",
    dtypes=[torch.float32],
    mode="eval",
)

# Save to file
out_path = "codefest/cf01/profiling/resnet18_profile.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(str(stats))

print(f"Profile saved to {out_path}")
print(f"\nQuick summary:")
print(f"  Total params      : {stats.total_params:,}")
print(f"  Trainable params  : {stats.trainable_params:,}")
print(f"  Total MACs        : {stats.total_mult_adds:,}")
print(f"  Total param mem   : {stats.total_param_bytes / 1e6:.2f} MB")
print(f"  Total output mem  : {stats.total_output_bytes / 1e6:.2f} MB")