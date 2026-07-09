import os
import subprocess
import sys

# --- CONFIGURATION ---
# Replace 'main.py' with the name of your entry point script
ENTRY_POINT = "GUI_tk.py"
EXE_NAME = "CatsEye"
ICON_PATH = "None"  # Set to None if you don't have an icon

# --- BUILD COMMAND CONSTRUCTION ---
cmd = [
    sys.executable,
    "-m",
    "nuitka",
    "--standalone",  # Produce a folder with the executable and all DLLs
    "--onefile",     # Pack everything into a single .exe file
    # "--windows-disable-console",  # Hide the cmd window (Uncomment for production!)
    "--enable-plugin=tk-inter",  # CRITICAL: Copies Tcl/Tk data files
    f"--output-filename={EXE_NAME}",
    ENTRY_POINT,
]

# Include icon if it exists
if ICON_PATH and os.path.exists(ICON_PATH):
    cmd.append(f"--windows-icon-from-ico={ICON_PATH}")


def run_build():
    print(f"🚀 Starting Nuitka compilation for {ENTRY_POINT}...")
    print(f"📋 Command: {' '.join(cmd)}\n")

    # Run the command and stream output to the term
    result = subprocess.run(cmd, check=True)
    if result.returncode == 0:
        print("\n✨ Build successful! Check your current directory for the .exe.")



if __name__ == "__main__":
    run_build()