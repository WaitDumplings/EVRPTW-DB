# Shortest-Path Oracle Benchmark

Setting: Cus1800/CS12, mother5000/CS120, one region, no plots, fixed seed 20260610.

| mode | instances | total s | s / instance | terminal matrix MB | failures |
|---|---:|---:|---:|---:|---:|
| source_cache | 1 | 32.761 | 32.761 | 0.0 | 0 |
| terminal_matrix | 1 | 46.771 | 46.771 | 100.0 | 0 |
| source_cache | 3 | 50.892 | 16.964 | 0.0 | 0 |
| terminal_matrix | 3 | 57.106 | 19.035 | 100.0 | 0 |
| source_cache | 10 | 130.090 | 13.009 | 0.0 | 0 |
| terminal_matrix | 10 | 135.294 | 13.529 | 100.0 | 0 |

## Takeaways

- `terminal_matrix` has a fixed precomputation cost: about 100 MB for 5121 terminals in this benchmark.
- For 1, 3, and 10 daily instances, `source_cache` is slightly faster in wall-clock time.
- The reason is that active-cluster sampling activates a subset of the mother board, so `terminal_matrix` over-computes distances for terminals not used by these few days.
- `terminal_matrix` is still useful when many days from the same region are needed immediately and memory is available; otherwise `source_cache` is the better default.

## Recommendation

Use `source_cache` as the default oracle mode for generation unless the region is reused many times and the terminal matrix precomputation can be amortized.
