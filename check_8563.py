import subprocess
import time
import os

kernel_id = "leventecsibi/vgh-qwen-8563"
# Use absolute path for kaggle.exe
kaggle_exe = r"C:\Users\csibi\Desktop\voicegenhub\.venv\Scripts\kaggle.exe"

def run_cmd(args, timeout=30):
    try:
        res = subprocess.run([kaggle_exe] + args, capture_output=True, text=True, timeout=timeout)
        return res.stdout, res.stderr
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"

print(f"Checking status for {kernel_id}...", flush=True)
stdout, stderr = run_cmd(["kernels", "status", kernel_id])
print(f"Status Output: {stdout}", flush=True)
if stderr: print(f"Status Error: {stderr}", flush=True)

print(f"Attempting to get output/logs for {kernel_id}...", flush=True)
# Try logs first just in case
stdout, stderr = run_cmd(["kernels", "logs", kernel_id])
if stdout:
    print("Logs command succeeded!")
    print(stdout)
else:
    print(f"Logs command failed/empty: {stderr}")

# Then try output
stdout, stderr = run_cmd(["kernels", "output", kernel_id, "-p", "latest_output"])
print(f"Output stdout: {stdout}", flush=True)
if stderr: print(f"Output stderr: {stderr}", flush=True)

if os.path.exists("latest_output"):
    print("Files in latest_output:", flush=True)
    for f in os.listdir("latest_output"):
        print(f" - {f}")
        if f.endswith(".log") or f.endswith(".txt") or f == "kaggle_remote.log":
            with open(os.path.join("latest_output", f), "r") as logf:
                lines = logf.readlines()
                print(f"Last 10 lines of {f}:")
                for line in lines[-10:]:
                    print(line.strip())
else:
    print("No output folder created.")
