"""Tests for rabbitllm.profiler."""

from rabbitllm.profiler import LayeredProfiler


def test_add_profiling_time_accumulates():
    p = LayeredProfiler(print_memory=False)
    p.add_profiling_time("layer_0", 0.1)
    p.add_profiling_time("layer_0", 0.2)
    assert p.profiling_time_dict["layer_0"] == [0.1, 0.2]


def test_clear_profiling_time_empties_dict():
    p = LayeredProfiler(print_memory=False)
    p.add_profiling_time("layer_0", 0.1)
    p.clear_profiling_time()
    assert p.profiling_time_dict["layer_0"] == []


def test_print_profiling_time_logs():
    p = LayeredProfiler(print_memory=False)
    p.add_profiling_time("layer_0", 0.1)
    p.add_profiling_time("layer_0", 0.2)
    p.print_profiling_time()
