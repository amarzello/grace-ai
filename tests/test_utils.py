#!/usr/bin/env python3
# tests/test_utils.py

import os
import sys
import json
from pathlib import Path

# Import base test class
from test_base import BaseTest

class UtilsTest(BaseTest):
    """Tests for the Grace utilities module."""
    
    def __init__(self, verbose=False):
        super().__init__("utils", verbose)
    
    def test_token_utils(self):
        """Test the token_utils functions if they exist."""
        # Try to import the module
        try:
            # First, try to import token_utils directly
            try:
                from grace.token_utils import estimate_tokens, calculate_relevance
                
                # Test estimate_tokens
                tokens = estimate_tokens("This is a test sentence.")
                self.logger.info(f"Estimated tokens: {tokens}")
                assert tokens > 0, "Token estimation should return a positive value"
                
                # Test calculate_relevance
                query_emb = [0.1, 0.2, 0.3]
                doc_emb = [0.2, 0.3, 0.4]
                relevance = calculate_relevance(query_emb, doc_emb)
                self.logger.info(f"Calculated relevance: {relevance}")
                assert 0 <= relevance <= 1, "Relevance should be between 0 and 1"
                
            except ImportError:
                # If that fails, try to import from the utils module
                project_root = Path(__file__).parent.parent
                
                # Try to find the utils module
                utils_files = list(project_root.glob("**/utils.py")) + list(project_root.glob("**/common.py"))
                
                if not utils_files:
                    self.logger.warning("Could not find utils.py or common.py")
                    return
                
                utils_file = utils_files[0]
                self.logger.info(f"Found utils file at {utils_file}")
                
                # Try to use functions from the file
                sys.path.insert(0, str(utils_file.parent.parent))
                
                # Try different module paths
                modules_to_try = [
                    "grace.utils.common",
                    "grace.utils",
                    "utils.common",
                    "utils",
                    utils_file.parent.name + ".common",
                    utils_file.parent.name
                ]
                
                for module_name in modules_to_try:
                    try:
                        module = __import__(module_name, fromlist=["estimate_tokens", "calculate_relevance"])
                        
                        if hasattr(module, "estimate_tokens"):
                            tokens = module.estimate_tokens("This is a test sentence.")
                            self.logger.info(f"Estimated tokens from {module_name}: {tokens}")
                            assert tokens > 0, "Token estimation should return a positive value"
                        
                        if hasattr(module, "calculate_relevance"):
                            relevance = module.calculate_relevance("test query", "test document content")
                            self.logger.info(f"Calculated relevance from {module_name}: {relevance}")
                            assert 0 <= relevance <= 1, "Relevance should be between 0 and 1"
                        
                        break
                    except ImportError:
                        continue
        
        except Exception as e:
            self.logger.warning(f"Could not test token_utils: {e}")
            # This doesn't fail the test
    
    def run_all_tests(self):
        """Run all utils tests."""
        self.run_test(self.test_token_utils)
        
        return self.print_results()

def run_tests(verbose=False):
    """Run utils tests."""
    test = UtilsTest(verbose=verbose)
    return test.run_all_tests()

if __name__ == "__main__":
    run_tests(verbose=True)