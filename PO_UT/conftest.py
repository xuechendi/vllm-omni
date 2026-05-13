"""Pytest configuration for WAN 2.2 test suite."""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))


def pytest_sessionfinish(session, exitstatus):
    """Generate report after all tests complete."""
    try:
        import test_wan22_kernels_ops_bf16

        if test_wan22_kernels_ops_bf16.TEST_RESULTS:
            test_wan22_kernels_ops_bf16.generate_markdown_report()
    except Exception as e:
        print(f"Warning: Could not generate report: {e}")
