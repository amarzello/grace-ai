#!/usr/bin/env python3#!/usr/bin/env python3
# tests/run_tests.py

import argparse
import importlib
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Available test modules
TEST_MODULES = {
    "basic": "test_basic",
    "utils": "test_utils"
}

def main():
    parser = argparse.ArgumentParser(description="Run Grace AI system tests")
    parser.add_argument("--component", choices=list(TEST_MODULES.keys()) + ["all"], 
                        default="all", help="Component to test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Set up environment for testing
    os.environ["GRACE_TESTING"] = "1"
    if args.debug:
        os.environ["GRACE_DEBUG"] = "1"
    
    # Determine which modules to test
    modules_to_test = []
    if args.component == "all":
        modules_to_test = list(TEST_MODULES.values())
    else:
        modules_to_test = [TEST_MODULES[args.component]]
    
    # Run the tests
    success = True
    for module_name in modules_to_test:
        try:
            print(f"\n{'='*60}\nRunning tests for: {module_name}\n{'='*60}")
            module = importlib.import_module(module_name)
            module_success = module.run_tests(verbose=args.verbose)
            if not module_success:
                success = False
        except Exception as e:
            print(f"Error running tests for {module_name}: {e}")
            success = False
    
    # Return status code
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())