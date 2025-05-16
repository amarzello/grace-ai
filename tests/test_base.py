#!/usr/bin/env python3
# tests/test_base.py

import os
import sys
import time
import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

class BaseTest:
    """Base class for Grace AI system tests."""
    
    def __init__(self, component_name: str, verbose: bool = False):
        self.component_name = component_name
        self.verbose = verbose
        self.logger = logging.getLogger(f"test.{component_name}")
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        
        # Set logging level based on verbose flag
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
    
    def run_test(self, test_func, *args, **kwargs):
        """Run a synchronous test function with proper error handling."""
        self.tests_run += 1
        test_name = test_func.__name__
        self.logger.info(f"Running test: {test_name}")
        
        start_time = time.time()
        try:
            test_func(*args, **kwargs)
            duration = time.time() - start_time
            self.logger.info(f"✓ Test {test_name} PASSED ({duration:.3f}s)")
            self.tests_passed += 1
            return True
        except AssertionError as e:
            duration = time.time() - start_time
            self.logger.error(f"✗ Test {test_name} FAILED ({duration:.3f}s): {e}")
            self.tests_failed += 1
            if self.verbose:
                import traceback
                self.logger.debug(traceback.format_exc())
            return False
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"✗ Test {test_name} ERROR ({duration:.3f}s): {e}")
            self.tests_failed += 1
            import traceback
            self.logger.debug(traceback.format_exc())
            return False
    
    def print_results(self):
        """Print test run results."""
        print(f"\nTest Results for {self.component_name}:")
        print(f"  Tests Run: {self.tests_run}")
        print(f"  Tests Passed: {self.tests_passed}")
        print(f"  Tests Failed: {self.tests_failed}")
        
        if self.tests_failed == 0:
            print(f"\n✓ All {self.tests_run} tests PASSED!")
            return True
        else:
            print(f"\n✗ {self.tests_failed} of {self.tests_run} tests FAILED.")
            return False

def run_async_tests(coro):
    """Helper function to run async tests."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)