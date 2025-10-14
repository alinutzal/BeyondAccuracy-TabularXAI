"""
Test suite for TabPFN PyTorch Lightning and Hydra integration.

Tests:
1. Lightning support in TabPFN module
2. Hydra configuration files for TabPFN
3. Configuration loading and parameter validation
4. API consistency with other models
"""

import os
import sys
import unittest
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from omegaconf import OmegaConf


class TestTabPFNLightningSupport(unittest.TestCase):
    """Test PyTorch Lightning support in TabPFN."""
    
    def test_lightning_imports(self):
        """Test that Lightning imports are present in tab_pfn.py."""
        tab_pfn_path = repo_root / 'src' / 'models' / 'tab_pfn.py'
        with open(tab_pfn_path, 'r') as f:
            content = f.read()
        
        self.assertIn('import lightning as L', content, 
                     "Lightning import not found")
        self.assertIn('LIGHTNING_AVAILABLE', content,
                     "LIGHTNING_AVAILABLE flag not found")
    
    def test_use_lightning_parameter(self):
        """Test that use_lightning parameter exists in TabPFNClassifier."""
        tab_pfn_path = repo_root / 'src' / 'models' / 'tab_pfn.py'
        with open(tab_pfn_path, 'r') as f:
            content = f.read()
        
        self.assertIn('use_lightning', content,
                     "use_lightning parameter not found")
        self.assertIn('self.use_lightning', content,
                     "use_lightning attribute not set")
    
    def test_lightning_warning_message(self):
        """Test that Lightning availability warning exists."""
        tab_pfn_path = repo_root / 'src' / 'models' / 'tab_pfn.py'
        with open(tab_pfn_path, 'r') as f:
            content = f.read()
        
        self.assertIn('PyTorch Lightning not available', content,
                     "Lightning warning message not found")


class TestTabPFNHydraConfigs(unittest.TestCase):
    """Test Hydra configuration files for TabPFN."""
    
    def test_config_files_exist(self):
        """Test that all TabPFN config files exist."""
        config_files = [
            'conf/model/tabpfn_default.yaml',
            'conf/model/tabpfn_fast.yaml',
            'conf/model/tabpfn_accurate.yaml'
        ]
        
        for config_file in config_files:
            config_path = repo_root / config_file
            self.assertTrue(config_path.exists(),
                          f"Config file not found: {config_file}")
    
    def test_config_structure(self):
        """Test that config files have correct structure."""
        config_files = [
            'tabpfn_default.yaml',
            'tabpfn_fast.yaml',
            'tabpfn_accurate.yaml'
        ]
        
        for config_file in config_files:
            config_path = repo_root / 'conf' / 'model' / config_file
            cfg = OmegaConf.load(config_path)
            
            # Check model name
            self.assertEqual(cfg.model.name, 'TabPFN',
                           f"Model name incorrect in {config_file}")
            
            # Check required parameters exist
            self.assertIn('device', cfg.model.params,
                        f"device parameter missing in {config_file}")
            self.assertIn('N_ensemble_configurations', cfg.model.params,
                        f"N_ensemble_configurations missing in {config_file}")
            self.assertIn('random_state', cfg.model.params,
                        f"random_state missing in {config_file}")
    
    def test_config_parameters(self):
        """Test that config parameters have correct values."""
        # Test default config
        cfg_default = OmegaConf.load(repo_root / 'conf' / 'model' / 'tabpfn_default.yaml')
        self.assertEqual(cfg_default.model.params.N_ensemble_configurations, 32,
                        "Default config should have 32 ensembles")
        
        # Test fast config
        cfg_fast = OmegaConf.load(repo_root / 'conf' / 'model' / 'tabpfn_fast.yaml')
        self.assertEqual(cfg_fast.model.params.N_ensemble_configurations, 8,
                        "Fast config should have 8 ensembles")
        
        # Test accurate config
        cfg_accurate = OmegaConf.load(repo_root / 'conf' / 'model' / 'tabpfn_accurate.yaml')
        self.assertEqual(cfg_accurate.model.params.N_ensemble_configurations, 64,
                        "Accurate config should have 64 ensembles")
    
    def test_device_parameter(self):
        """Test that all configs have device parameter."""
        config_files = [
            'tabpfn_default.yaml',
            'tabpfn_fast.yaml',
            'tabpfn_accurate.yaml'
        ]
        
        for config_file in config_files:
            config_path = repo_root / 'conf' / 'model' / config_file
            cfg = OmegaConf.load(config_path)
            
            self.assertEqual(cfg.model.params.device, 'cuda',
                           f"Device should be 'cuda' in {config_file}")
    
    def test_random_state_parameter(self):
        """Test that all configs have random_state parameter."""
        config_files = [
            'tabpfn_default.yaml',
            'tabpfn_fast.yaml',
            'tabpfn_accurate.yaml'
        ]
        
        for config_file in config_files:
            config_path = repo_root / 'conf' / 'model' / config_file
            cfg = OmegaConf.load(config_path)
            
            self.assertEqual(cfg.model.params.random_state, 42,
                           f"Random state should be 42 in {config_file}")


class TestTabPFNDocumentation(unittest.TestCase):
    """Test that documentation files exist and are updated."""
    
    def test_lightning_hydra_doc_exists(self):
        """Test that TABPFN_LIGHTNING_HYDRA.md exists."""
        doc_path = repo_root / 'readmd' / 'TABPFN_LIGHTNING_HYDRA.md'
        self.assertTrue(doc_path.exists(),
                       "TABPFN_LIGHTNING_HYDRA.md documentation not found")
    
    def test_hydra_usage_updated(self):
        """Test that HYDRA_USAGE.md mentions TabPFN."""
        usage_path = repo_root / 'HYDRA_USAGE.md'
        if usage_path.exists():
            with open(usage_path, 'r') as f:
                content = f.read()
            
            self.assertIn('TabPFN', content,
                         "HYDRA_USAGE.md should mention TabPFN")
            self.assertIn('tabpfn_default', content,
                         "HYDRA_USAGE.md should mention tabpfn_default config")
    
    def test_lightning_hydra_doc_content(self):
        """Test that TABPFN_LIGHTNING_HYDRA.md has required sections."""
        doc_path = repo_root / 'readmd' / 'TABPFN_LIGHTNING_HYDRA.md'
        with open(doc_path, 'r') as f:
            content = f.read()
        
        required_sections = [
            'Summary',
            'Changes Made',
            'PyTorch Lightning Support',
            'Hydra Configuration Files',
            'Usage',
            'Configuration Parameters'
        ]
        
        for section in required_sections:
            self.assertIn(section, content,
                        f"Documentation should have '{section}' section")


class TestTabPFNExamples(unittest.TestCase):
    """Test that example files exist."""
    
    def test_hydra_example_exists(self):
        """Test that tabpfn_hydra_example.py exists."""
        example_path = repo_root / 'examples' / 'tabpfn_hydra_example.py'
        self.assertTrue(example_path.exists(),
                       "tabpfn_hydra_example.py not found")
    
    def test_hydra_example_imports(self):
        """Test that hydra example has correct imports."""
        example_path = repo_root / 'examples' / 'tabpfn_hydra_example.py'
        with open(example_path, 'r') as f:
            content = f.read()
        
        self.assertIn('from omegaconf import OmegaConf', content,
                     "Example should import OmegaConf")
        self.assertIn('from models import TabPFNClassifier', content,
                     "Example should import TabPFNClassifier")
        self.assertIn('use_lightning', content,
                     "Example should demonstrate use_lightning parameter")


def run_tests():
    """Run all tests."""
    print("="*80)
    print("Running TabPFN Lightning and Hydra Integration Tests")
    print("="*80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTabPFNLightningSupport))
    suite.addTests(loader.loadTestsFromTestCase(TestTabPFNHydraConfigs))
    suite.addTests(loader.loadTestsFromTestCase(TestTabPFNDocumentation))
    suite.addTests(loader.loadTestsFromTestCase(TestTabPFNExamples))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(run_tests())
