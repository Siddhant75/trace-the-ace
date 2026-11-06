import os
import sys
import subprocess
import shutil

def download_outputs():
    out_dir = os.path.join("datasets", "deberta_outputs")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Downloading Kaggle kernel outputs to {out_dir}...")
    kernel_ref = os.environ["TRACE_ACE_KAGGLE_KERNEL"]
    cmd = ["kaggle", "kernels", "output", kernel_ref, "-p", out_dir]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    print("Return code:", res.returncode)
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)

    print("\n--- Downloaded Files in datasets/deberta_outputs/ ---")
    if os.path.exists(out_dir):
        files = os.listdir(out_dir)
        for f in files:
            fp = os.path.join(out_dir, f)
            size_mb = os.path.getsize(fp) / (1024 * 1024)
            print(f"File: {f} | Size: {size_mb:.2f} MB")
            
            # If it's a model checkpoint, copy to src/models/
            if f.startswith("deberta_fold_") and f.endswith(".pt"):
                dest = os.path.join("src", "models", f)
                shutil.copy(fp, dest)
                print(f" -> Copied checkpoint to {dest}")
            # If it's deberta_oof.csv, copy to datasets/
            if f == "deberta_oof.csv":
                dest = os.path.join("datasets", f)
                shutil.copy(fp, dest)
                print(f" -> Copied OOF to {dest}")

if __name__ == "__main__":
    download_outputs()
