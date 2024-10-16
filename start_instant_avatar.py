import subprocess
import sys
import time
import signal
import threading
import os

def start_service(command):
    return subprocess.Popen(
        command.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=dict(os.environ, PYTHONUNBUFFERED="1"),
    )

def print_output(process, service_name):
    def read_stream(stream):
        while True:
            line = stream.readline()
            if not line:
                break
            print(f"{service_name}: {line.rstrip()}", flush=True)

    stdout_thread = threading.Thread(target=read_stream, args=(process.stdout,), daemon=True)
    stderr_thread = threading.Thread(target=read_stream, args=(process.stderr,), daemon=True)

    stdout_thread.start()
    stderr_thread.start()

def main():
    # Start the API service
    api_service = start_service("python instant_id_api.py")
    print("Started instant_id_api.py")

    # Start a thread to print API service output
    print_output(api_service, "API")

    # Wait for the API service to initialize (adjust time as needed)
    time.sleep(10)  # Increased wait time to ensure API is fully initialized

    # Start the UI service
    ui_service = start_service("python instant_id_final_program.py")
    print("Started instant_id_final_program.py")

    # Start a thread to print UI service output
    print_output(ui_service, "UI")

    def signal_handler(sig, frame):
        print("Shutting down services...")
        api_service.terminate()
        ui_service.terminate()
        sys.exit(0)

    # Set up signal handling for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Wait for services to complete (or be interrupted)
    api_service.wait()
    ui_service.wait()

if __name__ == "__main__":
    main()