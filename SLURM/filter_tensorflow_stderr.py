#!/usr/bin/env python3
import re
import sys


SUPPRESSED_PATTERNS = (
    re.compile(r"Unable to register cu(?:FFT|DNN|BLAS) factory"),
    re.compile(
        r"All log messages before absl::InitializeLog\(\) is called are written to STDERR"
    ),
    re.compile(
        r"gpu_timer\.cc:\d+\] Skipping the delay kernel, measurement accuracy will be reduced"
    ),
    re.compile(r".*\(ignoring feature\)"),
)


def should_suppress(line: str) -> bool:
    return any(pattern.search(line) for pattern in SUPPRESSED_PATTERNS)


def main() -> None:
    for raw_line in sys.stdin:
        if should_suppress(raw_line):
            continue
        sys.stderr.write(raw_line)


if __name__ == "__main__":
    main()
