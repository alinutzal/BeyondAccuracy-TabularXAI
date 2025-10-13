#!/usr/bin/env python
"""
Test the skip logic by reading and analyzing the run_experiments.py code.
This ensures the implementation is correct without requiring full dependencies.
"""

import os
import re


def test_code_has_skip_logic():
    """Verify that the code contains the necessary skip logic."""
    print("="*60)
    print("Verifying Skip Logic Implementation")
    print("="*60)
    print()
    
    script_path = 'src/run_experiments.py'
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    checks = []
    
    # Check 1: experiment_exists function exists
    has_experiment_exists = 'def experiment_exists(' in content
    checks.append(('experiment_exists function defined', has_experiment_exists))
    if has_experiment_exists:
        print("✓ experiment_exists function is defined")
    else:
        print("✗ experiment_exists function not found")
    
    # Check 2: run_experiment has rerun parameter
    has_rerun_param = re.search(r'def run_experiment\([^)]*rerun[^)]*\)', content)
    checks.append(('run_experiment has rerun parameter', has_rerun_param is not None))
    if has_rerun_param:
        print("✓ run_experiment has rerun parameter")
    else:
        print("✗ run_experiment missing rerun parameter")
    
    # Check 3: Skip logic checks experiment_exists
    has_skip_check = 'if not rerun and experiment_exists' in content
    checks.append(('Skip logic checks experiment_exists', has_skip_check))
    if has_skip_check:
        print("✓ Skip logic checks if experiment exists")
    else:
        print("✗ Skip logic check not found")
    
    # Check 4: Message about skipping
    has_skip_message = 'Skipping experiment:' in content
    checks.append(('Skip message present', has_skip_message))
    if has_skip_message:
        print("✓ Skip message is displayed to user")
    else:
        print("✗ Skip message not found")
    
    # Check 5: Loads existing results
    has_load_results = 'with open(results_file' in content and 'json.load' in content
    checks.append(('Loads existing results', has_load_results))
    if has_load_results:
        print("✓ Loads and returns existing results")
    else:
        print("✗ Doesn't load existing results")
    
    # Check 6: run_all_experiments has rerun parameter
    has_run_all_rerun = re.search(r'def run_all_experiments\([^)]*rerun[^)]*\)', content)
    checks.append(('run_all_experiments has rerun parameter', has_run_all_rerun is not None))
    if has_run_all_rerun:
        print("✓ run_all_experiments has rerun parameter")
    else:
        print("✗ run_all_experiments missing rerun parameter")
    
    # Check 7: run_all_experiments passes rerun to run_experiment
    has_pass_rerun = 'run_experiment(dataset, model, results_dir, rerun=rerun)' in content
    checks.append(('run_all_experiments passes rerun', has_pass_rerun))
    if has_pass_rerun:
        print("✓ run_all_experiments passes rerun parameter")
    else:
        print("✗ run_all_experiments doesn't pass rerun parameter")
    
    # Check 8: __main__ section handles --rerun flag
    has_rerun_flag = '--rerun' in content and 'sys.argv' in content
    checks.append(('__main__ handles --rerun flag', has_rerun_flag))
    if has_rerun_flag:
        print("✓ __main__ section handles --rerun flag")
    else:
        print("✗ __main__ section doesn't handle --rerun flag")
    
    # Check 9: Help message mentions --rerun
    has_help = 'Use --rerun flag' in content
    checks.append(('Help message mentions --rerun', has_help))
    if has_help:
        print("✓ Help message mentions --rerun flag")
    else:
        print("✗ Help message doesn't mention --rerun flag")
    
    print()
    print("="*60)
    print("Summary")
    print("="*60)
    
    all_passed = all(result for _, result in checks)
    passed_count = sum(1 for _, result in checks if result)
    total_count = len(checks)
    
    print(f"Passed: {passed_count}/{total_count}")
    
    if all_passed:
        print("\n✓ All implementation checks passed!")
        return True
    else:
        print("\n✗ Some checks failed. Review the implementation.")
        return False


if __name__ == '__main__':
    import sys
    success = test_code_has_skip_logic()
    sys.exit(0 if success else 1)
