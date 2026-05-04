"""
Example 2: How yield actually executes — freeze and resume.

Key mental model:
  - Calling the function does NOT execute any code. It creates a frozen generator.
  - Each next() call RESUMES from where it paused, runs until the next yield, pauses again.
  - All local variables survive between pauses — the function's state is bookmarked.
"""

def get_words():
    print("  [inside function] About to yield 'hello'")
    yield "hello"
    print("  [inside function] Resumed! About to yield 'world'")
    yield "world"
    print("  [inside function] Resumed! About to yield 'bye'")
    yield "bye"
    print("  [inside function] No more yields — function ends")


if __name__ == "__main__":
    print("Step 0: Calling get_words() — does NOTHING run yet?")
    gen = get_words()
    print(f"  Got a generator object: {gen}")
    print(f"  (Notice: no '[inside function]' prints above — nothing ran!)\n")

    print("Step 1: First next() call")
    val = next(gen)
    print(f"  Received: '{val}'\n")

    print("Step 2: Second next() call")
    val = next(gen)
    print(f"  Received: '{val}'\n")

    print("Step 3: Third next() call")
    val = next(gen)
    print(f"  Received: '{val}'\n")

    print("Step 4: Fourth next() call — no more yields")
    try:
        val = next(gen)
    except StopIteration:
        print("  StopIteration raised! Generator is exhausted.")
