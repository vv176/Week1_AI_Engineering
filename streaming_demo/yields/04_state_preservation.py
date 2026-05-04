"""
Example 4: Yield preserves state — local variables survive between yields.

The generator remembers where it was AND what its variables were.
This works even with infinite generators (while True).
"""

# --- Finite generator with state ---
def countdown(n):
    print(f"  [countdown] Starting from {n}")
    while n > 0:
        yield n
        n -= 1                     # n survives between yields
        print(f"  [countdown] Resumed, n is now {n}")
    print("  [countdown] Done!")


# --- Infinite generator ---
def counter(start=1):
    n = start
    while True:                    # runs forever — produces values on demand
        yield n
        n += 1


if __name__ == "__main__":
    print("=== Countdown (finite, state preserved) ===")
    for val in countdown(5):
        print(f"  Got: {val}")

    print()

    print("=== Infinite counter (first 10 values) ===")
    gen = counter()
    for i in range(10):
        val = next(gen)
        print(f"  next() -> {val}")

    print(f"\n  Generator is NOT exhausted — we can keep going:")
    print(f"  next() -> {next(gen)}")     # 11
    print(f"  next() -> {next(gen)}")     # 12
