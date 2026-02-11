#!/usr/bin/env python3
"""
Segmented sieve -> SQLite prime table with rank as PRIMARY KEY.

Schema:
  primes(rank INTEGER PRIMARY KEY, prime INTEGER NOT NULL UNIQUE)

Default limit is 2^32 - 1 (all 32-bit primes).
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
import time
from typing import List, Tuple


U32_MAX = (1 << 32) - 1


def simple_sieve(n: int) -> List[int]:
    """Return all primes <= n using a simple sieve (for base primes up to sqrt(limit))."""
    if n < 2:
        return []
    bs = bytearray(b"\x01") * (n + 1)
    bs[0:2] = b"\x00\x00"
    r = int(math.isqrt(n))
    for p in range(2, r + 1):
        if bs[p]:
            step = p
            start = p * p
            bs[start : n + 1 : step] = b"\x00" * (((n - start) // step) + 1)
    return [i for i in range(2, n + 1) if bs[i]]


def segmented_primes(limit: int, segment_bytes: int) -> Tuple[List[int], int]:
    """
    Generator-like helper: yields primes in increasing order in segments.
    Returns (base_primes, sqrt_limit).
    """
    sqrt_limit = int(math.isqrt(limit))
    base = simple_sieve(sqrt_limit)
    return base, sqrt_limit


def create_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    # Speed-oriented pragmas (safe enough for bulk import; you can revert afterward)
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = NORMAL;")
    cur.execute("PRAGMA temp_store = MEMORY;")
    cur.execute("PRAGMA cache_size = -200000;")  # ~200MB cache (negative = KB)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS primes (
            rank  INTEGER PRIMARY KEY,
            prime INTEGER NOT NULL UNIQUE
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_primes_prime ON primes(prime);")
    conn.commit()


def get_next_rank(conn: sqlite3.Connection) -> int:
    cur = conn.cursor()
    cur.execute("SELECT COALESCE(MAX(rank), 0) FROM primes;")
    (mx,) = cur.fetchone()
    return int(mx) + 1


def insert_batch(conn: sqlite3.Connection, rows: List[Tuple[int, int]]) -> None:
    conn.executemany("INSERT INTO primes(rank, prime) VALUES (?, ?);", rows)


def run(limit: int, db_path: str, segment_size: int, commit_every: int, resume: bool) -> None:
    if limit < 2:
        raise ValueError("limit must be >= 2")

    conn = sqlite3.connect(db_path)
    try:
        create_db(conn)

        rank = get_next_rank(conn) if resume else 1
        if not resume:
            conn.execute("DELETE FROM primes;")
            conn.commit()

        base_primes, sqrt_limit = segmented_primes(limit, segment_size)

        t0 = time.time()
        inserted = 0

        # Handle prime 2 explicitly, then sieve only odds in segments
        if rank == 1:
            insert_batch(conn, [(1, 2)])
            conn.commit()
            rank = 2
            inserted += 1

        # We represent only odd numbers in [low, high] per segment.
        # Choose segment in terms of *odd count* so memory is about segment_size bytes.
        # One byte per odd candidate.
        odd_bytes = max(1, segment_size)
        odd_count = odd_bytes  # 1 byte per odd candidate
        span = odd_count * 2   # covers this many integers (odds only)

        low = 3
        # If resuming, you may want to start from the last prime; keeping it simple:
        # resume continues rank numbering only (doesn't skip already-inserted values).
        # For true resume-by-value, you'd query max(prime) and set low accordingly.

        batch: List[Tuple[int, int]] = []
        commit_counter = 0

        # Precompute base primes excluding 2 (we're doing odds)
        base_odds = [p for p in base_primes if p != 2]

        while low <= limit:

            high = min(limit, low + span - 1)
            if high < low:
                break
            # Ensure low/high are odd bounds
            if low % 2 == 0:
                low += 1
            if high % 2 == 0:
                high -= 1
            if high < low:
                break

            # Sieve array: index i corresponds to n = low + 2*i
            size = ((high - low) // 2) + 1
            is_prime = bytearray(b"\x01") * size

            for p in base_odds:
                p2 = p * p
                if p2 > high:
                    break

                # Find first multiple of p in [low, high] that is odd and >= p^2
                start = max(p2, ((low + p - 1) // p) * p)
                if start % 2 == 0:
                    start += p
                # Mark multiples: step is 2p (keeps odd multiples)
                step = 2 * p
                # Convert to index in is_prime
                idx0 = (start - low) // 2
                is_prime[idx0:size:step // 2] = b"\x00" * (((size - 1 - idx0) // (step // 2)) + 1)

            # Emit primes from this segment
            for i, flag in enumerate(is_prime):
                if flag:
                    n = low + 2 * i
                    # n fits in 32-bit if limit does; SQLite stores as 64-bit signed integer
                    batch.append((rank, n))
                    rank += 1
                    inserted += 1
                    commit_counter += 1

                    if commit_counter >= commit_every:
                        insert_batch(conn, batch)
                        conn.commit()
                        batch.clear()
                        commit_counter = 0

            # progress
            if inserted and (inserted % 1_000_000 == 0):
                dt = time.time() - t0
                rate = inserted / dt if dt > 0 else 0.0
                print(f"inserted={inserted:,} last_rank={rank-1:,} high={high:,} rate={rate:,.0f} primes/s")

            low = high + 2  # next odd after high

        # flush
        if batch:
            insert_batch(conn, batch)
            conn.commit()

        dt = time.time() - t0
        print(f"Done. Inserted {inserted:,} primes in {dt:.2f}s. Last rank={rank-1:,}")

    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="primes_u32.sqlite3", help="SQLite database path")
    ap.add_argument("--limit", type=int, default=U32_MAX, help="Upper bound (default 2^32-1)")
    ap.add_argument("--segment-bytes", type=int, default=8_000_000,
                    help="Bytes for one segment sieve array (odds only). Default ~8MB.")
    ap.add_argument("--commit-every", type=int, default=200_000,
                    help="Commit every N primes (bigger is faster, riskier).")
    ap.add_argument("--resume", action="store_true",
                    help="Resume rank numbering from existing DB (does NOT skip by prime value).")
    args = ap.parse_args()

    if args.limit > U32_MAX:
        print("Note: limit > 2^32-1, you said '32-bit primes' so this exceeds that.", file=sys.stderr)

    run(
        limit=args.limit,
        db_path=args.db,
        segment_size=args.segment_bytes,
        commit_every=args.commit_every,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
