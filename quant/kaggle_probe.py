"""Diagnose Kaggle T4 PyTorch compute-capability compatibility.

Prints torch version, CUDA build, device CC, and tries tiny FP16/BF16 matmuls.
Determines whether 'no kernel image for device' is the failure mode and which
dtype works.
"""
import torch
print(f"torch={torch.__version__}  cuda={torch.version.cuda}  avail={torch.cuda.is_available()}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print(f"device={torch.cuda.get_device_name()}  cc=sm_{cap[0]}{cap[1]}")
    print(f"compiled_ccs={torch.cuda.get_arch_list()}")
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        try:
            a = torch.randn(64, 64, device="cuda", dtype=dt)
            b = (a @ a).sum().item()
            print(f"  matmul {dt}: OK ({b:.1f})")
        except Exception as e:
            print(f"  matmul {dt}: FAIL ({type(e).__name__}: {str(e)[:120]})")
