# WAN 2.2 Operations Test Report

## Summary
| Metric | Value |
|--------|-------|
| Total Tests | 2 |
| Total TFLOPs | 0.20 |
| Avg Memory BW | 0.60 GB/s |

## Detailed Results

| Test Name | Latency (ms) | TFLOPs | Memory BW (GB/s) | Shape | Notes |
|-----------|--------------|--------|------------------|-------|-------|
| Conv1D Audio 80K | 100.577 | 0.01 | 0.82 | B=1, IC=1, OC=512, K=10 | None |
| Conv1D Audio Projector | 64.482 | 0.19 | 0.38 | B=4, IC=1280, OC=2560, K=3 | S2V: 4 calls, 92.66ms total |
