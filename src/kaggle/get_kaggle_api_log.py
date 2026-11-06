import os
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    user_name, kernel_name = os.environ["TRACE_ACE_KAGGLE_KERNEL"].split("/", 1)
    api = KaggleApi()
    api.authenticate()

    print(f"Fetching output for {user_name}/{kernel_name}...")
    res = api.kernels_output(user_name, kernel_name)

    print("Status:", res.get("status"))
    print("Error:", res.get("error"))
    print("\n--- LOG CONTENT ---")
    log_text = res.get("log", "")
    print(log_text if log_text else "No log text found.")

if __name__ == "__main__":
    main()
