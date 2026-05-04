"""
Example 1: return vs yield — the fundamental difference.

A regular function computes everything, holds it all in memory, returns at once.
A generator function produces values ONE AT A TIME, on demand.
"""

# --- Regular function (return) ---
def get_words_return():
    result = []
    result.append("hello")
    result.append("world")
    result.append("bye")
    return result                  # all 3 words built in memory, returned at once


# --- Generator function (yield) ---
def get_words_yield():
    yield "hello"                  # produce "hello", then freeze
    yield "world"                  # produce "world", then freeze
    yield "bye"                    # produce "bye", then freeze


if __name__ == "__main__":
    print("=== Using return ===")
    words = get_words_return()
    print(f"Type: {type(words)}")          # <class 'list'>
    print(f"Value: {words}")               # ['hello', 'world', 'bye']
    print(f"All in memory at once: {words}")

    print()

    print("=== Using yield ===")
    gen = get_words_yield()
    print(f"Type: {type(gen)}")            # <class 'generator'>
    print(f"Value: {gen}")                 # generator object (NOT a list)

    # To get values, we iterate:
    print("Iterating:")
    for word in gen:
        print(f"  Got: {word}")
