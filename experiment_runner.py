"""Main experiment runner: orchestrates multiple runs and saves results."""

import os
from pathlib import Path
from typing import List
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from experiment_config import get_config
from smw_experiment import SMWExperiment
from rl_experiment import RLExperiment
from trajectory import TrajectoryRecord


def run_single_experiment(method: str, config, run_id: int) -> List[TrajectoryRecord]:
    """
    Run a single experiment instance.
    
    Args:
        method: "smw" or "rl"
        config: ExperimentConfig
        run_id: which run (1-10)
    
    Returns:
        List of TrajectoryRecords
    """
    if method == "smw":
        experiment = SMWExperiment(config, run_id)
    elif method == "rl":
        experiment = RLExperiment(config, run_id)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"\n{'='*60}")
    print(f"Run {run_id}/{config.num_runs} - {method.upper()}")
    print(f"{'='*60}")
    
    records = experiment.run()
    return records


class ExperimentRunner:
    """Orchestrates complete experiment workflow."""
    
    def __init__(self, keyword: str, results_dir: str = "results"):
        """
        Initialize runner.
        
        Args:
            keyword: Experiment keyword (e.g., "smw_tfbind8")
            results_dir: Directory to save results
        """
        self.keyword = keyword
        self.config = get_config(keyword)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.all_records: List[TrajectoryRecord] = []
    
    def run(self, parallel: bool = True, n_workers: int = None) -> pd.DataFrame:
        """
        Run complete experiment (all runs).
        
        Args:
            parallel: Whether to run in parallel
            n_workers: Number of parallel workers (default: all available)
        
        Returns:
            DataFrame with all trajectory records
        """
        print(f"\n{'#'*60}")
        print(f"Running experiment: {self.keyword}")
        print(f"Method: {self.config.method}")
        print(f"Dataset: {self.config.dataset}")
        print(f"Runs: {self.config.num_runs}")
        print(f"Iterations per run: {self.config.smw_config.num_iterations if self.config.method == 'smw' else self.config.rl_config.num_iterations}")
        print(f"Parallel: {parallel}")
        print(f"{'#'*60}\n")
        
        if parallel:
            self._run_parallel(n_workers)
        else:
            self._run_sequential()
        
        # Convert to DataFrame
        records_dicts = [r.to_dict() for r in self.all_records]
        df = pd.DataFrame(records_dicts)
        
        return df
    
    def _run_sequential(self):
        """Run experiments sequentially."""
        for run_id in range(1, self.config.num_runs + 1):
            records = run_single_experiment(
                self.config.method,
                self.config,
                run_id
            )
            self.all_records.extend(records)
            print(f"✓ Run {run_id}: {len(records)} records")
    
    def _run_parallel(self, n_workers: int = None):
        """Run experiments in parallel."""
        if n_workers is None:
            n_workers = os.cpu_count() or 4
        
        n_workers = min(n_workers, self.config.num_runs)
        
        print(f"Using {n_workers} workers for parallel execution\n")
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all jobs
            futures = {}
            for run_id in range(1, self.config.num_runs + 1):
                future = executor.submit(
                    run_single_experiment,
                    self.config.method,
                    self.config,
                    run_id
                )
                futures[future] = run_id
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                run_id = futures[future]
                try:
                    records = future.result()
                    self.all_records.extend(records)
                    completed += 1
                    print(f"✓ Run {run_id}: {len(records)} records ({completed}/{self.config.num_runs})")
                except Exception as e:
                    print(f"✗ Run {run_id} failed: {e}")
                    raise
    
    def save_results(self, df: pd.DataFrame) -> str:
        """
        Save results to CSV.
        
        Args:
            df: DataFrame with trajectory records
        
        Returns:
            Path to saved CSV
        """
        filename = f"trajectory_{self.keyword}.csv"
        filepath = self.results_dir / filename
        
        df.to_csv(filepath, index=False)
        
        print(f"\n{'='*60}")
        print(f"Results saved to: {filepath}")
        print(f"Total records: {len(df)}")
        print(f"{'='*60}\n")
        
        # Print summary
        print("Summary:")
        print(f"  Runs: {df['run_id'].max()}")
        print(f"  Iterations per run: {df.groupby('run_id')['iteration'].max().mean():.1f}")
        print(f"  Avg records per run: {len(df) / df['run_id'].max():.1f}")
        print(f"\nOracle scores:")
        print(f"  Min: {df['oracle_score'].min():.4f}")
        print(f"  Max: {df['oracle_score'].max():.4f}")
        print(f"  Mean: {df['oracle_score'].mean():.4f}")
        print(f"\nHamming distances:")
        print(f"  Min: {df['min_hamming_distance'].min()}")
        print(f"  Max: {df['min_hamming_distance'].max():.1f}")
        print(f"  Mean: {df['min_hamming_distance'].mean():.1f}")
        
        return str(filepath)
    
    def run_and_save(self, parallel: bool = True, n_workers: int = None) -> pd.DataFrame:
        """
        Run experiment and save results.
        
        Args:
            parallel: Whether to run in parallel
            n_workers: Number of parallel workers
        
        Returns:
            DataFrame with results
        """
        start_time = time.time()
        
        df = self.run(parallel=parallel, n_workers=n_workers)
        self.save_results(df)
        
        elapsed = time.time() - start_time
        print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        
        return df


if __name__ == "__main__":
    # Example usage
    runner = ExperimentRunner(keyword="smw_tfbind8")
    df = runner.run_and_save(parallel=True)
    print(df.head())
