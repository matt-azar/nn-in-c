# Run this file in your shell to compile the project

instructions = """
Usage: python3 compile.py <mode>

Modes:
  debug    - Compile with -Wall -Wextra
  release  - Compile with -Ofast
"""

import sys
from subprocess import run, CalledProcessError

def main():
    if len(sys.argv) != 2:
        print(instructions)
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "debug":
        cmd1 = "gcc -o mnist_nn main.c nn.c mnist_loader.c -lm -Wall -Wextra"
    elif mode == "release":
        cmd1 = "gcc -o mnist_nn main.c nn.c mnist_loader.c -lm -Ofast"
    else:
        print(f"Unknown mode: {mode}\n")
        print(instructions)
        sys.exit(1)

    cmd2 = "./mnist_nn"

    try:
        print(f"Running: {cmd1}")
        run(cmd1, shell=True, check=True)
    except CalledProcessError as e:
        print(f"Compilation failed with error code {e.returncode}.")
        sys.exit(1)

    try:
        print(f"Running: {cmd2}")
        run(cmd2, shell=True, check=True)
    except CalledProcessError as e:
        print(f"Execution failed with error code {e.returncode}.")
        sys.exit(1)

if __name__ == "__main__":
    main()
