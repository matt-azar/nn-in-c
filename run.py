# Run this file in your shell to compile the project.
# Uses the zig compiler. 

instructions = """
Usage: python3 run.py <mode>
"""

import sys
from subprocess import run, CalledProcessError

def main():
    cmd = ".zig-out/bin/nn-project"
    try:
        print(f"Running: {cmd}")
        run(cmd, shell=True, check=True)
    except CalledProcessError as e:
        print(f"Compilation failed with error code {e.returncode}.")
        sys.exit(1)

if __name__ == "__main__":
    main()
