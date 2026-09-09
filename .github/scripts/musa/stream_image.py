"""Report compressed image transfer progress while streaming into Docker."""

import sys
import time


def main():
    started = reported = time.monotonic()
    total = 0
    while chunk := sys.stdin.buffer.read(1024 * 1024):
        sys.stdout.buffer.write(chunk)
        total += len(chunk)
        now = time.monotonic()
        if now - reported >= 30:
            print(
                f"Image transfer: {total / 1024**3:.2f} GiB in {now - started:.0f}s",
                file=sys.stderr,
                flush=True,
            )
            reported = now
    sys.stdout.buffer.flush()
    print(
        f"Image stream complete: {total / 1024**3:.2f} GiB "
        f"in {time.monotonic() - started:.0f}s",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()
