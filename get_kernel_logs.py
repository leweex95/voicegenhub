from kaggle_connector.jobs import JobManager
import time
import sys

kernel_id = "leventecsibi/vgh-qwen-8563"
manager = JobManager(kernel_id=kernel_id)

print(f"Polling logs for {kernel_id}...", flush=True)
last_logs = ""
for i in range(12): # try up to 2 minutes
    logs = manager.get_logs()
    if logs and logs != last_logs:
        new_text = logs[len(last_logs):] if last_logs and logs.startswith(last_logs) else logs
        if new_text.strip():
            print(f"New logs captured at step {i}:", flush=True)
            print(new_text, flush=True)
            last_logs = logs
    
    status = manager.get_status().lower()
    print(f"Status: {status}", flush=True)
    if "complete" in status or "error" in status:
        break
    time.sleep(10)
