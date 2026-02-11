import sqlite3
import random
import time
import math
import csv
import sympy
import gc
from typing import List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

max_prime_rank = 203280220
conn = sqlite3.connect("primes.db")
cursor = conn.cursor()

def naive_prime_verifier(n: int) -> bool:
    """
    Simple primality test: check divisibility up to sqrt(n).
    O(sqrt(n)) time, which is very slow for large n.
    """
    if n < 2:
        return False
    r = int(n ** 0.5)
    for i in range(2, r + 1):
        if n % i == 0:
            return False
    return True

def is_prime_linear(n: int) -> bool:
    """Intentionally slow: O(n) time, O(1) space."""
    if n < 2:
        return False
    for d in range(2, n):
        if n % d == 0:
            return False
    return True


def miller_rabin_verifier(n: int) -> bool:
    """
    Deterministic primality test for n < 2^64.
    O(log n)^3 worst-case time.
    """
    if n < 2:
        return False

    # Small primes (cheap filters)
    small_primes = (
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
        31, 37
    )
    if n in small_primes:
        return True
    for p in small_primes:
        if n % p == 0:
            return n == p

    # Write n−1 = d * 2^s
    d = n - 1
    s = 0
    while d & 1 == 0:
        d >>= 1
        s += 1

    # Deterministic Miller–Rabin bases for 64-bit ints
    # Proven sufficient set
    test_bases = (
        2, 325, 9375, 28178, 450775, 9780504, 1795265022
    )

    for a in test_bases:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False

    return True


def get_random_primes(count: int = 10_000) -> list[tuple[int, int]]:
    primes: list[tuple[int, int]] = []
    ranks = random.sample(range(1, max_prime_rank + 1), count)
    cursor.execute(
        "SELECT rank, prime FROM primes WHERE rank IN ({seq})".format(
            seq=",".join(["?"] * len(ranks))
        ),
        ranks,
    )
    for rank, p in cursor.fetchall():
        primes.append((rank, p))
    return primes


def per_prime_times(verifier, primes, warmup=200, use_cpu_time=True, repeat=25):
    """
    primes: list of (rank, prime) tuples
    Returns list of (rank, prime, time_ns)
    """
    items = list(primes)  # stable snapshot
    # warm up verifier (not timed)
    for _, p in items[:min(warmup, len(items))]:
        verifier(p)

    times = []
    clock = time.process_time_ns if use_cpu_time else time.perf_counter_ns

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for r, p in tqdm(items, desc=f"Verifying primes ({verifier.__name__})"):
            t0 = clock()
            ok = [verifier(p) for _ in range(repeat)]
            t1 = clock()
            if not ok:
                print(f"Verification failed for rank {r} with prime {p}")
            times.append((r, p, t1 - t0))
    finally:
        if gc_was_enabled:
            gc.enable()

    return times

def write_times(path, times):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "prime", "time_ns"])
        w.writerows(times)

if __name__ == "__main__":
    n = 1_000
    plot = True

    print(f"Fetching {n} random primes from the database...")
    primes = get_random_primes(n)  # dict rank -> prime
    print("Verifying primes...")

    print("Using naive verifier (O(sqrt(n)))...")
    times_naive = per_prime_times(naive_prime_verifier, primes)

    print("Using Miller-Rabin verifier (polylog)...")
    times_mr = per_prime_times(miller_rabin_verifier, primes)

    print("Using built-in sympy.isprime ...")
    times_sympy = per_prime_times(sympy.isprime, primes)

    print("Using linear verifier (O(n))...")
    times_linear = per_prime_times(is_prime_linear, primes[:100],warmup=10,repeat=5)  # only do for first 100 due to extreme slowness

    print("Writing results...")
    write_times("naive_times.csv", times_naive)
    write_times("miller_rabin_times.csv", times_mr)
    write_times("sympy_times.csv", times_sympy)
    write_times("linear_times.csv", times_linear)

    print("Done. Results written.")

    for name, times in [("naive", times_naive), ("miller_rabin", times_mr), ("sympy", times_sympy), ("linear", times_linear)]:

        print("Plotting results...")
        plt.figure()
        plt.scatter([p for r, p, t in times], [t for r, p, t in times], s=2)
        plt.xlabel("Time (ns)")
        plt.ylabel("Prime value")
        plt.title(f"Prime verification ({name}) time vs prime (time on x-axis)")
        plt.savefig(f"{name}_times.png")
