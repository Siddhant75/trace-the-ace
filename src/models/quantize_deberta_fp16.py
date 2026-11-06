"""
Quantize DeBERTa-v3-Large fold checkpoints from FP32 to FP16.
Reduces checkpoint file sizes from 828 MB each to ~414 MB each (total 4.14 GB -> 2.07 GB).
"""
import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "src", "models")

def quantize_checkpoints():
    print("=== Quantizing DeBERTa Checkpoints to FP16 ===")
    total_orig = 0
    total_fp16 = 0

    for fold in range(5):
        fp32_path = os.path.join(MODELS_DIR, f"deberta_fold_{fold}.pt")
        fp16_path = os.path.join(MODELS_DIR, f"deberta_fold_{fold}_fp16.pt")

        if not os.path.exists(fp32_path):
            print(f"Warning: {fp32_path} not found!")
            continue

        orig_sz = os.path.getsize(fp32_path) / (1024 * 1024)
        total_orig += orig_sz

        print(f"Loading {os.path.basename(fp32_path)} ({orig_sz:.1f} MB)...")
        sd = torch.load(fp32_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]

        # Convert floating point tensors to half precision (float16)
        sd_fp16 = {}
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                sd_fp16[k] = v.half()
            else:
                sd_fp16[k] = v

        torch.save(sd_fp16, fp16_path)
        fp16_sz = os.path.getsize(fp16_path) / (1024 * 1024)
        total_fp16 += fp16_sz
        print(f"  -> Saved {os.path.basename(fp16_path)}: {fp16_sz:.1f} MB (Saved {orig_sz - fp16_sz:.1f} MB)")

    print(f"\n=== Total Checkpoint Size Reduced from {total_orig:.1f} MB to {total_fp16:.1f} MB ({total_fp16/(1024):.2f} GB) ===")

if __name__ == "__main__":
    quantize_checkpoints()
