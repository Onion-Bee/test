import subprocess
import sys
import time
from pathlib import Path

def run_and_time(script_path):
    # Check if file exists
    script = Path(script_path)
    if not script.is_file():
        print(f"Error: File '{script_path}' not found.")
        sys.exit(1)

    # Record start time
    start_time = time.perf_counter()

    # Run the target script
    try:
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nScript exited with error code {e.returncode}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        # Record end time
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"\nExecution time: {elapsed:.4f} seconds")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_timer.py <script_to_run.py>")
        sys.exit(1)

    run_and_time(sys.argv[1])