"""
Example 5: The GIL — the one rule that decides whether threads help you.

CPython has a Global Interpreter Lock: only ONE thread runs Python bytecode
at a time, no matter how many cores you have.

So how did Examples 1-4 get faster? Because a thread that is WAITING is not
running bytecode. time.sleep(), a socket read, a DB round trip — the thread
releases the GIL and someone else runs. The waits overlap.

The rule:
  I/O-bound  (network, disk, API calls)  -> threads WIN.  Waits overlap.
  CPU-bound  (math, parsing, embeddings) -> threads DON'T. The GIL serialises.

For CPU-bound work you need separate processes, each with its own interpreter
and its own GIL: ProcessPoolExecutor. Same API, different cost model.

Almost everything in an agent — LLM calls, vector DB queries, HTTP tools,
file reads — is I/O. Which is why threads are the right default here.
"""

import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

N_TASKS = 4


def io_task(_):
    time.sleep(1)                          # waiting: GIL is RELEASED
    return "io done"


def cpu_task(_):
    total = 0
    for i in range(8_000_000):             # computing: GIL is HELD
        total += i * i
    return total


def timeit(label, fn):
    start = time.time()
    fn()
    elapsed = time.time() - start
    print(f"  {label:34} {elapsed:5.2f}s")
    return elapsed


if __name__ == "__main__":
    print("=== I/O-bound: 4 x sleep(1) ===")
    seq_io = timeit("sequential", lambda: [io_task(i) for i in range(N_TASKS)])
    thr_io = timeit("ThreadPoolExecutor(4)", lambda: list(
        ThreadPoolExecutor(4).map(io_task, range(N_TASKS))))
    print(f"  -> speedup: {seq_io / thr_io:.1f}x   threads help")

    print()

    print("=== CPU-bound: 4 x 8M-iteration loop ===")
    seq_cpu = timeit("sequential", lambda: [cpu_task(i) for i in range(N_TASKS)])
    thr_cpu = timeit("ThreadPoolExecutor(4)", lambda: list(
        ThreadPoolExecutor(4).map(cpu_task, range(N_TASKS))))
    print(f"  -> speedup: {seq_cpu / thr_cpu:.1f}x   threads do NOT help (GIL)")

    # Processes sidestep the GIL — one interpreter per process.
    # Needs the __main__ guard (which is why this whole file has one).
    with ProcessPoolExecutor(max_workers=4) as ex:
        proc_cpu = timeit("ProcessPoolExecutor(4)", lambda: list(
            ex.map(cpu_task, range(N_TASKS))))
    print(f"  -> speedup: {seq_cpu / proc_cpu:.1f}x   processes DO help")

    print()
    print("Diagnostic question before you reach for a thread pool:")
    print("  'Is this code waiting, or is it working?'")
    print("  Waiting -> threads.   Working -> processes.")
    print()
    print("Processes are not free: arguments and return values must be pickled")
    print("and copied between processes. For a 1ms task that overhead dominates.")
