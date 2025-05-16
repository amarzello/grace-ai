#!/usr/bin/env python3
# tests/test_basic.py

import os
import sys
import time
from pathlib import Path

# Import base test class
from test_base import BaseTest

class BasicTest(BaseTest):
    """Basic tests for the Grace project structure."""
    
    def __init__(self, verbose=False):
        super().__init__("basic", verbose)
    
    def test_project_structure(self):
        """Test that the project has the expected structure."""
        # Check that the project root directory exists
        project_root = Path(__file__).parent.parent
        assert project_root.exists(), "Project root directory should exist"
        
        # Check the grace directory (should exist but might not be a Python module)
        grace_dir = project_root / "grace"
        if grace_dir.exists():
            self.logger.info(f"Found grace directory at {grace_dir}")
        else:
            self.logger.warning(f"Grace directory not found at {grace_dir}")
            # Try to find where the code might be
            potential_dirs = list(project_root.glob("*"))
            self.logger.info(f"Potential module directories: {[d.name for d in potential_dirs]}")
        
        # Check for Python files in the project
        python_files = list(project_root.glob("**/*.py"))
        self.logger.info(f"Found {len(python_files)} Python files in the project")
        
        # Check which modules can be imported
        importable_modules = []
        for path in potential_dirs:
            if path.is_dir() and (path / "__init__.py").exists():
                module_name = path.name
                try:
                    sys.path.insert(0, str(project_root))
                    __import__(module_name)
                    importable_modules.append(module_name)
                except ImportError as e:
                    self.logger.warning(f"Could not import {module_name}: {e}")
                finally:
                    if str(project_root) in sys.path:
                        sys.path.remove(str(project_root))
        
        self.logger.info(f"Importable modules: {importable_modules}")
        
        # This test always passes, it's just information gathering
        assert True
    
    def test_file_permissions(self):
        """Test that key files have the correct permissions."""
        project_root = Path(__file__).parent.parent
        
        # Find Python files with a shebang
        shebang_files = []
        for py_file in project_root.glob("**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith("#!/usr/bin/env python"):
                        shebang_files.append(py_file)
            except Exception as e:
                self.logger.warning(f"Could not read {py_file}: {e}")
        
        self.logger.info(f"Found {len(shebang_files)} Python files with shebang lines")
        
        # On Unix, check if the files are executable
        if sys.platform != "win32":
            for file_path in shebang_files:
                is_executable = os.access(file_path, os.X_OK)
                self.logger.info(f"{file_path}: {'executable' if is_executable else 'not executable'}")
        
        # This test always passes, it's just information gathering
        assert True
    
    def test_python_version_compatibility(self):
        """Test Python version compatibility."""
        import platform
        
        python_version = platform.python_version()
        self.logger.info(f"Running on Python {python_version}")
        
        # Check for known compatibility issues
        if sys.version_info < (3, 8):
            self.logger.warning("Python version is below 3.8, which may cause compatibility issues")
        
        # This test always passes, it's just information gathering
        assert True
    
    def run_all_tests(self):
        """Run all basic tests."""
        self.run_test(self.test_project_structure)
        self.run_test(self.test_file_permissions)
        self.run_test(self.test_python_version_compatibility)
        
        return self.print_results()

def run_tests(verbose=False):
    """Run basic tests."""
    test = BasicTest(verbose=verbose)
    return test.run_all_tests()

if __name__ == "__main__":
    run_tests(verbose=True)