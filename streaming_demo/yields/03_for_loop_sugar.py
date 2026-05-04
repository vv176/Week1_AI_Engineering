"""
Example 3: The for loop is just syntactic sugar for next() + StopIteration.

These two blocks do EXACTLY the same thing:

  Block A (manual):
      gen = get_words()
      while True:
          try:
              val = next(gen)
              print(val)
          except StopIteration:
              break

  Block B (for loop):
      for val in get_words():
          print(val)
"""

def get_words():
    yield "hello"
    yield "world"
    yield "bye"


if __name__ == "__main__":
    print("=== Manual next() loop ===")
    gen = get_words()
    while True:
        try:
            val = next(gen)
            print(f"  Got: {val}")
        except StopIteration:
            print("  (StopIteration — done)")
            break

    print()

    print("=== For loop (same thing, cleaner) ===")
    for val in get_words():
        print(f"  Got: {val}")
    print("  (loop ended — done)")
