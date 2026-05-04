"""
Example 5: Why yield matters — memory usage comparison.

We generate a large dataset (1 million items) two ways:
  1. return: build entire list in memory, then process
  2. yield: produce one item at a time, constant memory

This is the production argument for generators.
"""

import sys

# --- Approach 1: return (all in memory) ---
def squares_list(n):
    result = []
    for i in range(n):
        result.append(i * i)
    return result


# --- Approach 2: yield (one at a time) ---
def squares_generator(n):
    for i in range(n):
        yield i * i


if __name__ == "__main__":
    N = 1_000_000

    print(f"Generating {N:,} squared numbers\n")

    # Approach 1: list
    data_list = squares_list(N)
    list_size = sys.getsizeof(data_list)
    print(f"List approach:")
    print(f"  Type: {type(data_list)}")
    print(f"  Memory: {list_size:,} bytes ({list_size / 1024 / 1024:.1f} MB)")
    print(f"  All {len(data_list):,} items in memory at once")

    print()

    # Approach 2: generator
    data_gen = squares_generator(N)
    gen_size = sys.getsizeof(data_gen)
    print(f"Generator approach:")
    print(f"  Type: {type(data_gen)}")
    print(f"  Memory: {gen_size:,} bytes ({gen_size} bytes!)")
    print(f"  Items produced on demand, one at a time")

    print()
    print(f"Memory ratio: list is {list_size / gen_size:.0f}x larger than generator")
    print()

    # Both produce the same results
    print("Verification — first 5 values match:")
    list_vals = squares_list(5)
    gen_vals = [next(squares_generator(5)) for _ in range(5)]
    # Actually let's do it properly
    gen = squares_generator(N)
    gen_first_5 = [next(gen) for _ in range(5)]
    print(f"  List:      {list_vals[:5]}")
    print(f"  Generator: {gen_first_5}")
