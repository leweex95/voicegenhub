
import sys
from pathlib import Path

# Add .venv site-packages to path just in case, though it's usually automatic
# But we'll use the .venv/Scripts/python.exe to run this.

from kaggle_connector.jobs import JobManager

def check_kernel(kernel_id, num_lines=50):
    try:
        manager = JobManager(kernel_id=kernel_id)
        status = manager.get_status()
        print(f"Status for {kernel_id}: {status}")
        
        print(f"\nLast logs (raw, truncated to avoid overflow):")
        logs = manager.get_logs()
        if logs:
            if isinstance(logs, str):
                # Sometimes it's a huge JSON string, let's try to parse it
                import json
                try:
                    data = json.loads(logs)
                    if isinstance(data, list):
                        for entry in data[-num_lines:]:
                            prefix = f"[{entry.get('stream_name')}]"
                            content = entry.get('data')
                            print(f"{prefix} {content}", end="")
                    else:
                        print(str(data)[-2000:])
                except:
                    print(logs[-2000:])
            else:
                print(str(logs)[-2000:])
        else:
            print("No logs available yet.")
            
    except Exception as e:
        print(f"Error checking kernel {kernel_id}: {e}")

if __name__ == "__main__":
    kernel_id = "leventecsibi/vgh-qwen-5964"
    num_lines = 50
    if len(sys.argv) > 1:
        kernel_id = sys.argv[1]
    if len(sys.argv) > 2:
        num_lines = int(sys.argv[2])
    check_kernel(kernel_id, num_lines)
