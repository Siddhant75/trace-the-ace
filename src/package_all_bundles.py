"""
Package Generator for Both Competition Submission Bundles:
1. submission_EXP7_dual_stream.zip (Bundle B)
2. submission_EXP8_all_levers.zip (Bundle C)
3. submission.zip (Active Default)
"""
import os
import zipfile
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
BUILD_DIR = os.path.join(BASE_DIR, "submission_package")

def create_bundle_zip(out_zip_filename, description):
    out_zip_path = os.path.join(BASE_DIR, out_zip_filename)
    if os.path.exists(out_zip_path):
        os.remove(out_zip_path)

    with zipfile.ZipFile(out_zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(BUILD_DIR):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, BUILD_DIR)
                if rel_path.startswith("data") or rel_path.startswith("submission.csv") or "bge-large-en-v1.5" in rel_path:
                    continue
                z.write(abs_path, rel_path)

    size_mb = os.path.getsize(out_zip_path) / (1024 * 1024)
    print(f"=== Successfully Created {out_zip_filename} ({size_mb:.2f} MB) - {description} ===")
    return out_zip_path

def main():
    print(f"=== Packaging Submission Bundles in {BASE_DIR} ===")

    # 1. Active EXP-8 Bundle C
    zip_exp8 = create_bundle_zip("submission_EXP8_all_levers.zip", "EXP-8 All 4 Levers (Student GroupKFold + Ratio Interactions + T-Scaling)")

    # 2. Main default submission.zip pointing to EXP-8
    shutil.copy(zip_exp8, os.path.join(BASE_DIR, "submission.zip"))
    print(f"=== Updated default submission.zip -> EXP-8 ===")

    print("\n--- Summary of Available Submission Archives ---")
    for fname in ["submission.zip", "submission_EXP8_all_levers.zip"]:
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.exists(fpath):
            print(f" - Path: {fpath} ({os.path.getsize(fpath)/(1024*1024):.2f} MB)")

if __name__ == "__main__":
    main()
