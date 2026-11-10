"""Print a colorized test summary and exit non-zero on any failure.

Reads the summary log line by line and prints each line with an ANSI color
based on its content:

* lines containing ``FAILED``   -> red   (and force a non-zero exit code)
* lines containing ``PASSED``   -> green
* lines containing ``NO_TESTS`` -> yellow
* everything else               -> unchanged
"""

import argparse
import sys

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def print_summary(summary_log: str) -> int:
    """Print the colorized summary and return the exit status."""
    status = 0
    with open(summary_log, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "FAILED" in line:
                print(f"{RED}{line}{RESET}")
                status = 1
            elif "PASSED" in line:
                print(f"{GREEN}{line}{RESET}")
            elif "NO_TESTS" in line:
                print(f"{YELLOW}{line}{RESET}")
            else:
                print(line)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-log", required=True, help="Path to the summary log file.")
    args = parser.parse_args()
    sys.exit(print_summary(args.summary_log))


if __name__ == "__main__":
    main()
