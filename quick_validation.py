"""Quick validation script - checks imports and basic structure without heavy computation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("\n" + "="*60)
print("QUICK VALIDATION - Experiment Framework")
print("="*60)

# TEST 1: Imports
print("\n✓ Testing imports...")
try:
    from trajectory import TrajectoryRecord
    from experiment_config import ExperimentConfig, get_config, EXPERIMENT_CONFIGS
    from base_experiment import BaseExperiment
    from smw_experiment import SMWExperiment
    from rl_experiment import RLExperiment
    from experiment_runner import ExperimentRunner
    print("  ✓ All modules imported successfully")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# TEST 2: Config validation
print("\n✓ Testing configurations...")
try:
    keywords = ['smw_tfbind8', 'smw_gb1', 'rl_tfbind8', 'rl_gb1']
    for keyword in keywords:
        config = get_config(keyword)
        assert config.method in ['smw', 'rl'], f"Invalid method: {config.method}"
        assert config.dataset in ['tfbind8', 'gb1'], f"Invalid dataset: {config.dataset}"
        assert config.num_runs == 10, f"num_runs should be 10, got {config.num_runs}"
        
        # Check iteration count
        if config.method == 'smw':
            assert config.smw_config.num_iterations == 20
        else:
            assert config.rl_config.num_iterations == 20
    
    print(f"  ✓ All {len(keywords)} configurations valid")
except Exception as e:
    print(f"  ✗ Config validation failed: {e}")
    sys.exit(1)

# TEST 3: TrajectoryRecord
print("\n✓ Testing TrajectoryRecord...")
try:
    record = TrajectoryRecord(
        sequence="ACGTACGT",
        oracle_score=0.75,
        surrogate_score=0.72,
        method="smw",
        iteration=0,
        run_id=1,
        seed=42,
        min_hamming_distance=5,
        dataset="tfbind8",
        transcription_factor="SIX6_REF_R1",
    )
    
    record_dict = record.to_dict()
    expected_keys = {
        'method', 'run_id', 'seed', 'iteration', 'sequence',
        'oracle_score', 'surrogate_score', 'min_hamming_distance',
        'dataset', 'transcription_factor'
    }
    
    assert set(record_dict.keys()) == expected_keys, f"Missing keys: {expected_keys - set(record_dict.keys())}"
    assert record.method == "smw"
    assert record.oracle_score == 0.75
    assert record.min_hamming_distance == 5
    
    print(f"  ✓ TrajectoryRecord working correctly")
except Exception as e:
    print(f"  ✗ TrajectoryRecord test failed: {e}")
    sys.exit(1)

# TEST 4: Runner instantiation
print("\n✓ Testing ExperimentRunner...")
try:
    runner = ExperimentRunner(keyword="smw_tfbind8", results_dir="results_test")
    assert runner.config.method == "smw"
    assert runner.config.dataset == "tfbind8"
    assert runner.keyword == "smw_tfbind8"
    
    runner_rl = ExperimentRunner(keyword="rl_gb1")
    assert runner_rl.config.method == "rl"
    assert runner_rl.config.dataset == "gb1"
    
    print(f"  ✓ ExperimentRunner initialization working")
except Exception as e:
    print(f"  ✗ ExperimentRunner test failed: {e}")
    sys.exit(1)

# TEST 5: Check file structure
print("\n✓ Checking file structure...")
try:
    files_to_check = [
        "trajectory.py",
        "experiment_config.py",
        "base_experiment.py",
        "smw_experiment.py",
        "rl_experiment.py",
        "experiment_runner.py",
        "assets/compute_metrics.py",
    ]
    
    for f in files_to_check:
        path = Path(__file__).parent / f
        assert path.exists(), f"Missing file: {f}"
    
    print(f"  ✓ All required files present")
except Exception as e:
    print(f"  ✗ File check failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL VALIDATIONS PASSED!")
print("="*60)
print("\nFramework is ready. Structure summary:")
print("  - trajectory.py: TrajectoryRecord dataclass")
print("  - experiment_config.py: Config system (SMWConfig, RLConfig)")
print("  - base_experiment.py: Abstract base class (BaseExperiment)")
print("  - smw_experiment.py: SMW implementation")
print("  - rl_experiment.py: RL implementation")
print("  - experiment_runner.py: Main orchestrator")
print("  - assets/compute_metrics.py: Utility functions")

print("\nTo run experiments:")
print("  from experiment_runner import ExperimentRunner")
print("  runner = ExperimentRunner(keyword='smw_tfbind8')")
print("  df = runner.run_and_save(parallel=True)")

print("\nAvailable keywords:")
for kw in ['smw_tfbind8', 'smw_gb1', 'rl_tfbind8', 'rl_gb1']:
    config = get_config(kw)
    iters = config.smw_config.num_iterations if config.method == 'smw' else config.rl_config.num_iterations
    print(f"  - {kw:15s} ({config.method.upper():3s}, {iters} iterations)")
