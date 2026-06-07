import os
import sys
import subprocess
import time
import webbrowser

def launch_dashboard():
    print("==================================================")
    print("   GLIOMA MRI RETRIEVAL WEB DASHBOARD RUNNER      ")
    print("==================================================")
    
    python_exe = r"C:\gpu_env_311\Scripts\python.exe"
    if not os.path.exists(python_exe):
        print(f"Error: GPU python environment not found at {python_exe}")
        sys.exit(1)
        
    app_script = os.path.join("src", "web", "app.py")
    if not os.path.exists(app_script):
        print(f"Error: app.py not found at {app_script}")
        sys.exit(1)
        
    print("\n>>> Launching Flask Web Server...")
    print(">>> URL: http://127.0.0.1:5000")
    print(">>> Press CTRL+C to terminate server.\n")
    
    # Automatically open local browser page after a 1.5 seconds delay
    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:5000")
        
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run server
    try:
        subprocess.run([python_exe, app_script], check=True)
    except KeyboardInterrupt:
        print("\n>>> Dashboard Server terminated successfully.")
    except Exception as e:
        print(f"\n>>> Error running server: {e}")

if __name__ == "__main__":
    launch_dashboard()
