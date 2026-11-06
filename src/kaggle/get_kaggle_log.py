import os
import sys
import subprocess

def fetch_log():
    kernel_ref = os.environ["TRACE_ACE_KAGGLE_KERNEL"]
    cmd = ["kaggle", "kernels", "output", kernel_ref, "-p", "kaggle_out"]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)

    if os.path.exists("kaggle_out"):
        for f in os.listdir("kaggle_out"):
            fp = os.path.join("kaggle_out", f)
            print(f"\n=== File: {f} ({os.path.getsize(fp)} bytes) ===")
            if f.endswith(".log") or f.endswith(".txt"):
                with open(fp, "r", encoding="utf-8", errors="replace") as fh:
                    print(fh.read()[-3000:]) # Last 3000 characters

if __name__ == "__main__":
    fetch_log()
