"""
Example 6: Production use case — processing a large file with yield.

Step 1: Generate a sample file (100,000 lines of fake transaction data).
Step 2: Process it WITHOUT yield (load all lines into memory).
Step 3: Process it WITH yield (one line at a time, constant memory).

This is how real data pipelines handle files that don't fit in memory:
  - Log analysis (Splunk, ELK)
  - ETL pipelines (Airflow, Spark)
  - Database exports (pg_dump processing)
  - Kafka consumer patterns
"""

import os
import sys
import time
import random

SAMPLE_FILE = "sample_transactions.csv"
NUM_LINES = 100_000


def generate_sample_file():
    """Create a fake transactions CSV."""
    categories = ["food", "transport", "rent", "entertainment", "utilities", "shopping"]
    print(f"Generating {NUM_LINES:,} transaction records...")
    with open(SAMPLE_FILE, "w") as f:
        f.write("id,amount,category\n")
        for i in range(NUM_LINES):
            amount = round(random.uniform(10, 5000), 2)
            cat = random.choice(categories)
            f.write(f"{i},{amount},{cat}\n")
    size = os.path.getsize(SAMPLE_FILE)
    print(f"File created: {size / 1024:.0f} KB\n")


# --- WITHOUT yield: load everything ---
def read_all_lines(filepath):
    with open(filepath) as f:
        return f.readlines()           # entire file in memory as a list


# --- WITH yield: one line at a time ---
def read_lines(filepath):
    with open(filepath) as f:
        for line in f:
            yield line                 # one line, then forget it


# --- Processing function: sum amounts for a given category ---
def total_for_category_list(filepath, target_category):
    """Load all lines, then filter and sum."""
    lines = read_all_lines(filepath)
    total = 0.0
    for line in lines[1:]:             # skip header
        parts = line.strip().split(",")
        if parts[2] == target_category:
            total += float(parts[1])
    return total, sys.getsizeof(lines)


def total_for_category_generator(filepath, target_category):
    """Yield one line at a time, filter and sum."""
    gen = read_lines(filepath)
    next(gen)                          # skip header
    total = 0.0
    for line in gen:
        parts = line.strip().split(",")
        if parts[2] == target_category:
            total += float(parts[1])
    return total


if __name__ == "__main__":
    # Generate sample data
    generate_sample_file()

    target = "food"

    # Approach 1: list
    print(f"=== Without yield (load all lines) ===")
    start = time.time()
    total1, mem_used = total_for_category_list(SAMPLE_FILE, target)
    elapsed1 = time.time() - start
    print(f"  Total '{target}' spend: Rs {total1:,.2f}")
    print(f"  Memory for lines list: {mem_used:,} bytes ({mem_used / 1024:.0f} KB)")
    print(f"  Time: {elapsed1:.3f}s")

    print()

    # Approach 2: generator
    print(f"=== With yield (one line at a time) ===")
    start = time.time()
    total2 = total_for_category_generator(SAMPLE_FILE, target)
    elapsed2 = time.time() - start
    gen_obj_size = sys.getsizeof(read_lines(SAMPLE_FILE))
    print(f"  Total '{target}' spend: Rs {total2:,.2f}")
    print(f"  Memory for generator: {gen_obj_size} bytes")
    print(f"  Time: {elapsed2:.3f}s")

    print()
    print(f"Same result: {abs(total1 - total2) < 0.01}")
    print(f"Memory difference: list used ~{mem_used // gen_obj_size}x more memory")

    # Cleanup
    os.remove(SAMPLE_FILE)
    print(f"\nCleaned up {SAMPLE_FILE}")
