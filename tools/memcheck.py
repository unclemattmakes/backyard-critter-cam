"""memcheck.py -- a userspace RAM check for the rig's host. Not a substitute for MemTest86.

WHY THIS IS THE SECOND-BEST TOOL, AND WHY IT STILL EARNS ITS PLACE
------------------------------------------------------------------
MemTest86 is the right tool. It boots outside the OS, so it owns every byte of RAM including what
Windows reserves for itself, and it controls caching so it can be certain a read came from a DRAM
cell rather than from L3. A process running under Windows can do neither. It also costs a USB
stick, a reboot, and a human standing at the machine to choose a boot device -- and on this host a
reboot has a second cost, because the GPU power cap does not survive one unless
tools/apply_gpu_cap.ps1 has registered its boot task.

So this is the cheap first pass. It takes most of the free memory and hammers it with the patterns
that catch the failures actually worth catching:

  * stuck bits            -- all-zeros, all-ones, 0x55/0xAA/0x0F alternating
  * address-decode faults -- every word holds its own index, so a word that answers to the wrong
                             address is caught by value. These are among the most common real
                             failures, and a naive "write X, read X back" test cannot see them.
  * retention             -- with --soak, data is left sitting before it is read back, which is
                             what catches a weak cell that holds its charge for a second but not
                             for a minute.

READ THE RESULT HONESTLY
------------------------
A FAILURE here is conclusive: the RAM is bad, and you can stop testing and start swapping DIMMs.
A CLEAN RUN IS NOT A CLEAN BILL OF HEALTH. It cannot see memory held by the kernel, other
processes, or firmware; it cannot defeat the cache hierarchy; and it will not catch marginal
timing that only shows up hot, or disturbance effects between physically adjacent rows. It
reduces suspicion. It does not clear it. Run MemTest86 before concluding the memory is fine.

USAGE
    python tools/memcheck.py                      # auto-size, 2 passes
    python tools/memcheck.py --gib 32 --passes 4  # explicit size, more passes
    python tools/memcheck.py --soak 60            # leave data sitting 60s before reading it back
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

WORDS_PER_BLOCK = 1 << 24          # 16 Mi words = 128 MiB, small enough that comparison
                                   # temporaries stay trivial next to the buffer itself


def log(msg: str, fh) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  {msg}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    fh.write(line + "\n")
    fh.flush()
    os.fsync(fh.fileno())          # a host that hangs mid-test must still leave the log behind


def _blocks(n: int):
    for i in range(0, n, WORDS_PER_BLOCK):
        yield i, min(i + WORDS_PER_BLOCK, n)


def _report(arr, lo, expected, fh, budget: int) -> int:
    """Compare one block; print up to `budget` mismatches. Returns how many were found."""
    mism = np.nonzero(arr[lo:lo + expected.size] != expected)[0]
    for j in mism[:budget]:
        idx = lo + int(j)
        log(f"    *** MISMATCH at word {idx} (byte offset {idx * 8}): "
            f"expected 0x{int(expected[j]):016x}, read 0x{int(arr[idx]):016x}", fh)
    return int(mism.size)


def scalar_pattern(value: int):
    v = np.uint64(value)

    def fill(arr):
        arr[:] = v

    def check(arr, fh):
        bad = 0
        for lo, hi in _blocks(arr.size):
            exp = np.full(hi - lo, v, dtype=np.uint64)
            bad += _report(arr, lo, exp, fh, 8 if bad < 40 else 0)
        return bad

    return fill, check


def address_pattern():
    """Each word stores its own index -- a word that answers to the wrong address shows up."""

    def fill(arr):
        for lo, hi in _blocks(arr.size):
            arr[lo:hi] = np.arange(lo, hi, dtype=np.uint64)

    def check(arr, fh):
        bad = 0
        for lo, hi in _blocks(arr.size):
            bad += _report(arr, lo, np.arange(lo, hi, dtype=np.uint64), fh, 8 if bad < 40 else 0)
        return bad

    return fill, check


def random_pattern(seed: int):
    """Deterministic stream, so the expected value is regenerated rather than stored."""

    def fill(arr):
        rng = np.random.default_rng(seed)
        for lo, hi in _blocks(arr.size):
            arr[lo:hi] = rng.integers(0, 1 << 63, size=hi - lo, dtype=np.uint64)

    def check(arr, fh):
        rng = np.random.default_rng(seed)
        bad = 0
        for lo, hi in _blocks(arr.size):
            exp = rng.integers(0, 1 << 63, size=hi - lo, dtype=np.uint64)
            bad += _report(arr, lo, exp, fh, 8 if bad < 40 else 0)
        return bad

    return fill, check


def main() -> int:
    ap = argparse.ArgumentParser(description="Userspace RAM check (see module docstring).")
    ap.add_argument("--gib", type=float, default=0.0,
                    help="GiB to test (default: available memory minus --headroom)")
    ap.add_argument("--headroom", type=float, default=14.0,
                    help="GiB to leave free for the OS and the rig (default 14)")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--soak", type=int, default=0,
                    help="seconds to leave each pattern sitting before reading it back")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logpath = args.log or os.path.join(root, "logs", "memcheck.log")
    os.makedirs(os.path.dirname(logpath), exist_ok=True)

    with open(logpath, "a", encoding="utf-8") as fh:
        gib = args.gib
        if gib <= 0:
            try:
                import psutil
                gib = max(1.0, psutil.virtual_memory().available / (1 << 30) - args.headroom)
            except Exception:
                gib = 8.0
        words = int(gib * (1 << 30)) // 8
        gib = words * 8 / (1 << 30)

        log("=" * 78, fh)
        log(f"memcheck starting: {gib:.1f} GiB, {args.passes} pass(es), soak {args.soak}s", fh)
        log("NOTE: a clean run reduces suspicion but does NOT clear the RAM -- see the docstring.", fh)

        try:
            arr = np.empty(words, dtype=np.uint64)
        except MemoryError:
            log(f"FAILED to allocate {gib:.1f} GiB -- rerun with a smaller --gib.", fh)
            return 2

        patterns = [
            ("all-zeros      ", scalar_pattern(0x0000000000000000)),
            ("all-ones       ", scalar_pattern(0xFFFFFFFFFFFFFFFF)),
            ("0x5555...      ", scalar_pattern(0x5555555555555555)),
            ("0xAAAA...      ", scalar_pattern(0xAAAAAAAAAAAAAAAA)),
            ("0x0F0F...      ", scalar_pattern(0x0F0F0F0F0F0F0F0F)),
            ("address-in-addr", address_pattern()),
        ]

        total_bad = 0
        t_start = time.time()
        for p in range(1, args.passes + 1):
            run = patterns + [(f"random(seed={p})", random_pattern(p))]
            for name, (fill, check) in run:
                t0 = time.time()
                fill(arr)
                t_w = time.time() - t0
                if args.soak:
                    time.sleep(args.soak)
                t0 = time.time()
                bad = check(arr, fh)
                t_r = time.time() - t0
                total_bad += bad
                verdict = "OK" if bad == 0 else f"{bad} BAD WORD(S)"
                log(f"pass {p}/{args.passes}  {name}  write {gib / max(t_w, 1e-6):6.1f} GiB/s  "
                    f"verify {gib / max(t_r, 1e-6):6.1f} GiB/s  -> {verdict}", fh)

        mins = (time.time() - t_start) / 60
        log(f"memcheck finished in {mins:.1f} min -- "
            + (f"*** {total_bad} FAILED WORD(S): this RAM is bad ***" if total_bad
               else "no errors found (not a clean bill of health -- run MemTest86)"), fh)
        return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main())
