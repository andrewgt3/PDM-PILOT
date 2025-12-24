import subprocess
import shlex
import sys
import time
import os
import signal

def run_command(command, cwd=None, background=False):
    """Runs a command safely without shell=True.
    
    Args:
        command: Command string or list of arguments
        cwd: Working directory
        background: If True, run in background and return Popen object
    """
    # Convert string command to list of arguments (secure parsing)
    if isinstance(command, str):
        args = shlex.split(command)
    else:
        args = command
    
    if background:
        return subprocess.Popen(args, cwd=cwd, preexec_fn=os.setsid)  # nosec B103
    else:
        return subprocess.run(args, cwd=cwd)  # nosec B603

def cleanup(processes):
    """Kills all background processes."""
    print("\n🛑 Shutting down PlantAGI Demo...")
    for p in processes:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass  # Process already terminated
        except Exception as e:
            print(f"Warning: Could not terminate process: {e}")
    print("✓ Shutdown complete.")

def main():
    print("=" * 60)
    print("🏭 PlantAGI Enterprise Command - Demo Launcher")
    print("=" * 60)
    
    # 1. Cleanup existing instances
    print("Cleaning up old processes...")
    subprocess.run(["pkill", "-f", "streamlit run dashboard_streamlit.py"])  # nosec B603
    subprocess.run(["pkill", "-f", "mock_fleet_streamer.py"])  # nosec B603
    time.sleep(1)

    processes = []
    
    try:
        # 2. Start Data Simulator (Chaos Monkey)
        print("🚀 Starting Fleet Simulator & Chaos Monkey...")
        p_sim = run_command("python3 mock_fleet_streamer.py", background=True)
        processes.append(p_sim)
        time.sleep(2) # Wait for DB connection
        
        # 3. Start Dashboard
        print("📊 Launching Enterprise Dashboard...")
        p_dash = run_command("streamlit run dashboard_streamlit.py --server.port 8501", background=True)
        processes.append(p_dash)
        
        print("\n✅ DEMO IS LIVE!")
        print("   -> Dashboard: http://localhost:8501")
        print("   -> Simulator: Active (45s Healthy / 30s Failure Cycle)")
        print("\n[Press Ctrl+C to Stop]")
        
        p_dash.wait()
        
    except KeyboardInterrupt:
        cleanup(processes)
    except Exception as e:
        print(f"Error: {e}")
        cleanup(processes)

if __name__ == "__main__":
    main()
