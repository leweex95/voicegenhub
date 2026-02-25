from kaggle_connector.jobs import JobManager
import os
import sys

kernel_id = "leventecsibi/vgh-qwen-8563"
manager = JobManager(kernel_id=kernel_id)

print(f"Status: {manager.get_status()}", flush=True)
print(f"Fetching logs for {kernel_id}...", flush=True)
logs = manager.get_logs()
if logs:
    lines = logs.splitlines()
    print("Latest 10 lines of logs:")
    for line in lines[-10:]:
        print(line)
else:
    print("No logs yet. The kernel might still be in setup phase and hasn't produced a .log file yet.")
