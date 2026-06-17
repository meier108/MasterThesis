import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _build_default_breakpoints():
	fine_region = np.round(np.arange(3.0, 4.0 + 0.1, 0.1), 1).tolist()
	return [0.0, 1.0, 2.0, 2.5] + fine_region + [4.5, 5.0, 6.0, 7.0]


def _transform_values(values, breakpoints):
	"""Map y-values onto a stretched axis using piecewise-linear interpolation."""
	axis_positions = np.arange(len(breakpoints), dtype=float)
	values = np.asarray(values, dtype=float)
	return np.interp(values, breakpoints, axis_positions)


def _format_tick_label(value):
	if abs(value - int(value)) < 1e-9:
		return str(int(value))
	return f"{value:.1f}".rstrip("0").rstrip(".")


def plot_trajectories(
	trajectories_df,
	test_set,
	run=None,
	aggregate_runs=True,
	show_std=True,
	y_breakpoints=None,
	ax=None,
):
	"""Plot oracle and surrogate trajectories with non-uniform y-axis spacing.

	Args:
		trajectories_df: DataFrame containing at least columns
			["round", "oracle_score", "surrogate_score"].
		test_set: Value to filter by column "evaluated_split" (or "split").
		run: Optional run id if multiple runs are present.
		aggregate_runs: If True and run is None, aggregate all runs by round
			using mean and standard deviation.
		show_std: If True, draw +/- 1 std as a shaded band.
		y_breakpoints: Optional ascending y values that define axis density.
			If None, defaults to dense ticks between 3.0 and 4.0.
		ax: Optional matplotlib axis.
	"""
	if y_breakpoints is None:
		y_breakpoints = _build_default_breakpoints()

	y_breakpoints = np.asarray(y_breakpoints, dtype=float)
	if y_breakpoints.ndim != 1 or len(y_breakpoints) < 2:
		raise ValueError("y_breakpoints must be a 1D sequence with at least 2 values.")
	if np.any(np.diff(y_breakpoints) <= 0):
		raise ValueError("y_breakpoints must be strictly increasing.")

	split_col = "evaluated_split" if "evaluated_split" in trajectories_df.columns else "split"
	if split_col not in trajectories_df.columns:
		raise ValueError("trajectories_df must contain either 'evaluated_split' or 'split'.")

	df = trajectories_df[trajectories_df[split_col] == test_set].copy()

	if run is not None and "run" in df.columns:
		df = df[df["run"] == run]
	elif run is not None:
		raise ValueError("run was provided, but trajectories_df has no 'run' column.")

	if df.empty:
		raise ValueError(f"No trajectory rows found for test_set='{test_set}' and run='{run}'.")

	if ax is None:
		_, ax = plt.subplots(figsize=(9, 5.5))

	if aggregate_runs and run is None and "run" in df.columns:
		agg = (
			df.groupby("round", as_index=False)
			.agg(
				oracle_mean=("oracle_score", "mean"),
				oracle_std=("oracle_score", "std"),
				surrogate_mean=("surrogate_score", "mean"),
				surrogate_std=("surrogate_score", "std"),
			)
			.sort_values("round")
		)

		x = agg["round"].to_numpy(dtype=float)
		oracle_y = agg["oracle_mean"].to_numpy(dtype=float)
		surrogate_y = agg["surrogate_mean"].to_numpy(dtype=float)
		oracle_std = np.nan_to_num(agg["oracle_std"].to_numpy(dtype=float), nan=0.0)
		surrogate_std = np.nan_to_num(agg["surrogate_std"].to_numpy(dtype=float), nan=0.0)
		oracle_label = "Oracle mean"
		surrogate_label = "Surrogate mean"
	else:
		df = df.sort_values("round")
		x = df["round"].to_numpy(dtype=float)
		oracle_y = df["oracle_score"].to_numpy(dtype=float)
		surrogate_y = df["surrogate_score"].to_numpy(dtype=float)
		oracle_std = np.zeros_like(oracle_y)
		surrogate_std = np.zeros_like(surrogate_y)
		oracle_label = "Oracle score"
		surrogate_label = "Surrogate score"

	# Extend breakpoints if data fall outside configured range.
	oracle_low = oracle_y - oracle_std
	oracle_high = oracle_y + oracle_std
	surrogate_low = surrogate_y - surrogate_std
	surrogate_high = surrogate_y + surrogate_std

	data_min = float(min(np.min(oracle_low), np.min(surrogate_low)))
	data_max = float(max(np.max(oracle_high), np.max(surrogate_high)))
	if data_min < y_breakpoints[0]:
		y_breakpoints = np.insert(y_breakpoints, 0, data_min)
	if data_max > y_breakpoints[-1]:
		y_breakpoints = np.append(y_breakpoints, data_max)

	oracle_plot_y = _transform_values(oracle_y, y_breakpoints)
	surrogate_plot_y = _transform_values(surrogate_y, y_breakpoints)
	oracle_low_plot_y = _transform_values(oracle_low, y_breakpoints)
	oracle_high_plot_y = _transform_values(oracle_high, y_breakpoints)
	surrogate_low_plot_y = _transform_values(surrogate_low, y_breakpoints)
	surrogate_high_plot_y = _transform_values(surrogate_high, y_breakpoints)

	ax.plot(x, oracle_plot_y, color="tab:orange", marker="o", linewidth=2, label=oracle_label)
	ax.plot(
		x,
		surrogate_plot_y,
		color="tab:blue",
		marker="o",
		linewidth=2,
		linestyle="--",
		label=surrogate_label,
	)

	if show_std and (aggregate_runs and run is None and "run" in df.columns):
		ax.fill_between(
			x,
			oracle_low_plot_y,
			oracle_high_plot_y,
			color="tab:orange",
			alpha=0.2,
			label="Oracle +/- 1 std",
		)
		ax.fill_between(
			x,
			surrogate_low_plot_y,
			surrogate_high_plot_y,
			color="tab:blue",
			alpha=0.2,
			label="Surrogate +/- 1 std",
		)

	tick_positions = np.arange(len(y_breakpoints), dtype=float)
	tick_labels = [_format_tick_label(value) for value in y_breakpoints]

	ax.set_yticks(tick_positions)
	ax.set_yticklabels(tick_labels)
	ax.set_xlabel("Round")
	ax.set_ylabel("Score")
	title = f"Trajectory Scores for {test_set}"
	if aggregate_runs and run is None and "run" in df.columns:
		title += " (mean +/- std over runs)"
	elif run is not None:
		title += f" (run {run})"
	ax.set_title(title)
	ax.grid(True, linestyle=":", alpha=0.45)
	ax.legend()
	ax.set_xlim(x.min(), x.max())

	# Keep a small margin above/below the visible points.
	min_plot_y = float(min(np.min(oracle_low_plot_y), np.min(surrogate_low_plot_y)))
	max_plot_y = float(max(np.max(oracle_high_plot_y), np.max(surrogate_high_plot_y)))
	ax.set_ylim(min_plot_y - 0.3, max_plot_y + 0.3)

	return ax


def plot_metrics_vs_hamming_distance(
	df,
	target_col='oracle_score',
	rf_pred_col='rf_prediction',
	mlp_pred_col='mlp_prediction',
	hamming_col='min_hamming_distance',
	ax=None,
):
	"""Plot mean absolute error vs minimum Hamming distance for RF and MLP models.
	
	Args:
		df: DataFrame containing oracle scores, model predictions, and min hamming distance
		target_col: Column name for oracle/true scores (default: 'oracle_score')
		rf_pred_col: Column name for RF predictions (default: 'rf_prediction')
		mlp_pred_col: Column name for MLP predictions (default: 'mlp_prediction')
		hamming_col: Column name for minimum hamming distance (default: 'min_hamming_distance')
		ax: Optional matplotlib axis
	
	Returns:
		The matplotlib axis object
	"""
	
	# Calculate absolute errors
	df_temp = df.copy()
	df_temp['error_rf'] = np.abs(df_temp[target_col] - df_temp[rf_pred_col])
	df_temp['error_mlp'] = np.abs(df_temp[target_col] - df_temp[mlp_pred_col])
	
	# Group by hamming distance and calculate mean and std
	grouped = df_temp.groupby(hamming_col).agg(
		mean_error_rf=('error_rf', 'mean'),
		std_error_rf=('error_rf', 'std'),
		mean_error_mlp=('error_mlp', 'mean'),
		std_error_mlp=('error_mlp', 'std'),
	).reset_index()
	
	x = grouped[hamming_col]
	
	if ax is None:
		_, ax = plt.subplots(figsize=(12, 6))
	
	# Plot RF metrics with error band
	ax.plot(x, grouped['mean_error_rf'], label='RF vs Oracle', color='tab:blue', linewidth=2)
	ax.fill_between(
		x,
		grouped['mean_error_rf'] - grouped['std_error_rf'],
		grouped['mean_error_rf'] + grouped['std_error_rf'],
		color='tab:blue',
		alpha=0.2
	)
	
	# Plot MLP metrics with error band
	ax.plot(x, grouped['mean_error_mlp'], label='MLP vs Oracle', color='tab:orange', linewidth=2)
	ax.fill_between(
		x,
		grouped['mean_error_mlp'] - grouped['std_error_mlp'],
		grouped['mean_error_mlp'] + grouped['std_error_mlp'],
		color='tab:orange',
		alpha=0.2
	)
	
	ax.set_title('Mean Absolute Error vs Min Hamming Distance')
	ax.set_xlabel('Min Hamming Distance to Training Set')
	ax.set_ylabel('Mean Absolute Error')
	ax.legend()
	ax.grid(alpha=0.3)
	
	return ax

def plot_trajectory_quality_vs_distance(
	df,
	oracle_col='oracle_score',
	hamming_col='min_hamming_distance',
	plot_type='scatter',
	ax=None,
):
	"""Plot sequence quality vs minimum Hamming distance to assess strategy effectiveness.
	
	Shows whether your strategy still finds high-quality sequences as you move further from training set.
	
	Args:
		df: Trajectory DataFrame with oracle scores and hamming distances
		oracle_col: Column name for oracle scores (default: 'oracle_score')
		hamming_col: Column name for minimum hamming distance (default: 'min_hamming_distance')
		plot_type: 'scatter' (default) for individual points with mean line,
				   'box' for boxplots by distance, 'both' for combined view
		ax: Optional matplotlib axis
	
	Returns:
		The matplotlib axis object
	"""
	
	if ax is None:
		_, ax = plt.subplots(figsize=(12, 6))
	
	if plot_type in ['scatter', 'both']:
		# Scatter plot with transparency to see density
		ax.scatter(df[hamming_col], df[oracle_col], alpha=0.5, s=50, color='tab:blue', label='Individual sequences')
		
		# Add mean line
		grouped_mean = df.groupby(hamming_col)[oracle_col].mean().sort_index()
		ax.plot(grouped_mean.index, grouped_mean.values, color='tab:red', linewidth=2.5, marker='o', label='Mean quality', zorder=5)
	
	if plot_type in ['box', 'both']:
		# Create boxplot by grouping hamming distances
		hamming_distances = sorted(df[hamming_col].unique())
		data_by_distance = [df[df[hamming_col] == d][oracle_col].values for d in hamming_distances]
		
		if plot_type == 'box':
			# If only box plot, create from scratch
			ax.boxplot(data_by_distance, labels=hamming_distances, patch_artist=True)
			for patch in ax.artists:
				patch.set_facecolor('tab:blue')
				patch.set_alpha(0.7)
		else:
			# Add boxplot overlay on scatter
			bp = ax.boxplot(data_by_distance, positions=hamming_distances, widths=0.3, 
						   patch_artist=True, showfliers=False)
			for patch in bp['boxes']:
				patch.set_facecolor('tab:orange')
				patch.set_alpha(0.3)
	
	ax.set_xlabel('Min Hamming Distance to Training Set')
	ax.set_ylabel('Oracle Score (Sequence Quality)')
	ax.set_title('Strategy Effectiveness: Sequence Quality vs Distance from Training Set')
	ax.grid(alpha=0.3, axis='y')
	ax.legend()
	
	return ax


def plot_strategy_comparison_topk(
	dataframes_dict,
	oracle_col='oracle_score',
	k_values=[1, 5, 10, 25, 50],
	ax=None,
):
	"""Compare top-k sequence quality across different optimization strategies.
	
	Shows how different strategies compare in finding high-quality sequences.
	For each k value, computes mean of top-k sequences for each strategy.
	
	Args:
		dataframes_dict: Dict with strategy names as keys and DataFrames as values
						 (e.g., {'SMW': df_smw, 'RL': df_rl, 'GFlow': df_gflow})
		oracle_col: Column name for oracle scores (default: 'oracle_score')
		k_values: List of k values to evaluate (default: [1, 5, 10, 25, 50])
		ax: Optional matplotlib axis
	
	Returns:
		The matplotlib axis object

	TODO Usage: 
	from assets.plot import plot_strategy_comparison_topk
	import pandas as pd

	# Load trajectory data from each strategy
	df_smw = pd.read_csv('results/trajectory_smw_tfbind8.csv')
	df_rl = pd.read_csv('results/trajectory_rl_tfbind8.csv')
	df_gfn = pd.read_csv('results/trajectory_gfn_tfbind8.csv')

	# Compare strategies
	plot_strategy_comparison_topk({
		'SMW (Naive)': df_smw,
		'RL': df_rl,
		'GFlow Net': df_gfn
	})

	plt.show()
	"""
	
	if ax is None:
		_, ax = plt.subplots(figsize=(12, 6))
	
	# Compute top-k mean scores for each strategy
	results = {}
	for strategy_name, df in dataframes_dict.items():
		topk_means = []
		for k in k_values:
			top_k_scores = df.nlargest(k, oracle_col)[oracle_col]
			topk_means.append(top_k_scores.mean())
		results[strategy_name] = topk_means
	
	# Plot each strategy
	for strategy_name, means in results.items():
		ax.plot(k_values, means, marker='o', linewidth=2.5, markersize=8, label=strategy_name)
	
	ax.set_xlabel('Top-k Sequences', fontsize=12)
	ax.set_ylabel('Mean Oracle Score', fontsize=12)
	ax.set_title('Strategy Effectiveness: Top-k Sequence Quality Comparison', fontsize=13)
	ax.set_xticks(k_values)
	ax.grid(alpha=0.3)
	ax.legend(fontsize=11)
	
	return ax


def plot_surrogate_reliability_vs_distance(
	df,
	oracle_col='oracle_score',
	surrogate_col='surrogate_score',
	hamming_col='min_hamming_distance',
	metric='error',
	ax=None,
):
	"""Plot surrogate model reliability degradation as function of OOD distance.
	
	Shows whether surrogate models become less reliable (higher error/lower correlation)
	as sequences move further from the training set.
	
	Args:
		df: DataFrame with oracle scores, surrogate scores, and hamming distances
		oracle_col: Column name for oracle scores (default: 'oracle_score')
		surrogate_col: Column name for surrogate scores (default: 'surrogate_score')
		hamming_col: Column name for minimum hamming distance (default: 'min_hamming_distance')
		metric: 'error' for absolute error (default), 'mse' for squared error, 'correlation' for Spearman
		ax: Optional matplotlib axis
	
	Returns:
		The matplotlib axis object
	"""
	from scipy.stats import spearmanr
	
	if ax is None:
		_, ax = plt.subplots(figsize=(12, 6))
	
	# Calculate metric by hamming distance
	grouped_data = []
	hamming_distances = sorted(df[hamming_col].unique())
	
	for dist in hamming_distances:
		subset = df[df[hamming_col] == dist]
		
		if metric == 'error':
			errors = np.abs(subset[oracle_col] - subset[surrogate_col])
			metric_mean = errors.mean()
			metric_std = errors.std()
		elif metric == 'mse':
			errors = (subset[oracle_col] - subset[surrogate_col]) ** 2
			metric_mean = errors.mean()
			metric_std = errors.std()
		elif metric == 'correlation':
			if len(subset) > 1:
				corr = spearmanr(subset[oracle_col], subset[surrogate_col]).correlation
				metric_mean = corr
				metric_std = 0
			else:
				metric_mean = np.nan
				metric_std = 0
		
		grouped_data.append({'dist': dist, 'mean': metric_mean, 'std': metric_std})
	
	grouped_df = pd.DataFrame(grouped_data)
	
	# Plot with error bands
	ax.plot(grouped_df['dist'], grouped_df['mean'], 
			marker='o', linewidth=2.5, markersize=8, color='tab:red', label=f'{metric.title()} Mean')
	ax.fill_between(grouped_df['dist'], 
					grouped_df['mean'] - grouped_df['std'],
					grouped_df['mean'] + grouped_df['std'],
					color='tab:red', alpha=0.2, label='±1 Std Dev')
	
	# Add scatter points to show individual sample density
	ax.scatter(df[hamming_col], 
			  np.abs(df[oracle_col] - df[surrogate_col]) if metric == 'error' else 
			  (df[oracle_col] - df[surrogate_col]) ** 2 if metric == 'mse' else
			  df[oracle_col],
			  alpha=0.2, s=30, color='gray', label='Individual samples')
	
	ax.set_xlabel('Min Hamming Distance to Training Set', fontsize=12)
	if metric == 'error':
		ax.set_ylabel('Mean Absolute Error (Oracle vs Surrogate)', fontsize=12)
		ax.set_title('Surrogate Model Reliability: Prediction Error vs OOD Distance', fontsize=13)
	elif metric == 'mse':
		ax.set_ylabel('Mean Squared Error', fontsize=12)
		ax.set_title('Surrogate Model Reliability: MSE vs OOD Distance', fontsize=13)
	else:
		ax.set_ylabel('Spearman Correlation', fontsize=12)
		ax.set_title('Surrogate Model Reliability: Correlation vs OOD Distance', fontsize=13)
	
	ax.grid(alpha=0.3)
	ax.legend(fontsize=10)
	
	return ax


def plot_trajectory_optimization_progress(
	df,
	iteration_col='iteration',
	oracle_col='oracle_score',
	surrogate_col='surrogate_score',
	sequence_col='sequence',
	run_id_col='run_id',
	run_id=1,
	figsize=(15, 6),
	ax=None,
):
	"""Plot optimization progress with best sequences and mean scores by iteration.
	
	Creates a two-panel visualization:
	- Left: Scatter of oracle scores per iteration, colored by surrogate score, with best sequences annotated
	- Right: Mean oracle and surrogate scores over iterations with error bands
	
	Args:
		df: Trajectory DataFrame
		iteration_col: Column name for iteration (default: 'iteration')
		oracle_col: Column name for oracle scores (default: 'oracle_score')
		surrogate_col: Column name for surrogate scores (default: 'surrogate_score')
		sequence_col: Column name for sequences (default: 'sequence')
		run_id_col: Column name for run IDs (default: 'run_id')
		run_id: Specific run to plot (default: 1)
		figsize: Figure size (default: (15, 6))
		ax: Optional matplotlib axis (ignored, creates subplots)
	
	Returns:
		Tuple of (fig, axes)
	"""
	
	# Filter for specific run
	plot_df = df.where(df[run_id_col] == run_id).dropna(subset=[run_id_col]).copy()
	plot_df = plot_df.sort_values([iteration_col, oracle_col], ascending=[True, False]).reset_index(drop=True)
	
	# Get best sequence per iteration
	best_per_iteration = plot_df.loc[plot_df.groupby(iteration_col)[oracle_col].idxmax()].copy()
	
	fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1.4, 1]})
	
	# Left panel: Scatter plot with best sequences annotated
	scatter = axes[0].scatter(
		plot_df[iteration_col],
		plot_df[oracle_col],
		c=plot_df[surrogate_col],
		cmap='viridis',
		s=70,
		alpha=0.85,
		edgecolors='black',
		linewidths=0.5,
	)
	
	# Annotate best sequences
	for _, row in best_per_iteration.iterrows():
		axes[0].annotate(
			row[sequence_col],
			(row[iteration_col], row[oracle_col]),
			textcoords="offset points",
			xytext=(6, 8),
			fontsize=8,
		)
	
	axes[0].set_xlabel('Iteration', fontsize=12)
	axes[0].set_ylabel('Oracle Score', fontsize=12)
	axes[0].set_xticks(sorted(plot_df[iteration_col].unique()))
	axes[0].grid(True, alpha=0.3)
	cbar = plt.colorbar(scatter, ax=axes[0])
	cbar.set_label('Surrogate Score', fontsize=12)
	cbar.ax.tick_params(labelsize=10)
	
	# Right panel: Mean scores per iteration with error bands
	mean_scores = plot_df.groupby(iteration_col)[[oracle_col, surrogate_col]].mean().reset_index()
	std_scores = plot_df.groupby(iteration_col)[[oracle_col, surrogate_col]].std().reset_index()
	
	axes[1].plot(mean_scores[iteration_col], mean_scores[oracle_col], label='Oracle Score', linewidth=2)
	axes[1].plot(mean_scores[iteration_col], mean_scores[surrogate_col], label='Surrogate Score', linewidth=2)
	axes[1].fill_between(
		mean_scores[iteration_col],
		mean_scores[oracle_col] - std_scores[oracle_col],
		mean_scores[oracle_col] + std_scores[oracle_col],
		alpha=0.2,
	)
	axes[1].fill_between(
		mean_scores[iteration_col],
		mean_scores[surrogate_col] - std_scores[surrogate_col],
		mean_scores[surrogate_col] + std_scores[surrogate_col],
		alpha=0.2,
	)
	axes[1].set_xlabel('Iteration', fontsize=12)
	axes[1].set_ylabel('Mean Score', fontsize=12)
	axes[1].legend()
	axes[1].grid(True, alpha=0.3)
	
	plt.tight_layout()

	return fig, axes


def _compute_topk_curve(df, k, oracle_col, iteration_col, run_col):
	"""Return (iterations, means, stds) for cumulative top-k mean per iteration."""
	max_iter = int(df[iteration_col].max())
	iterations = list(range(1, max_iter + 1))
	run_ids = df[run_col].unique()
	per_cutoff_per_run = np.full((len(iterations), len(run_ids)), np.nan)
	for ri, run_id in enumerate(run_ids):
		run_df = df[df[run_col] == run_id]
		for ci, cutoff in enumerate(iterations):
			subset = run_df[run_df[iteration_col] <= cutoff]
			if len(subset) == 0:
				continue
			top_k = subset.nlargest(min(k, len(subset)), oracle_col)[oracle_col]
			per_cutoff_per_run[ci, ri] = top_k.mean()
	means = np.nanmean(per_cutoff_per_run, axis=1)
	stds = np.nanstd(per_cutoff_per_run, axis=1)
	return iterations, means, stds


def plot_topk_over_iterations(
	dataframes_dict,
	k=10,
	oracle_col='oracle_score',
	iteration_col='iteration',
	run_col='run_id',
):
	"""Plot mean top-k oracle score as iterations accumulate, one subplot per strategy.

	Each method gets its own panel so wildly different score scales (e.g. SMW on
	GB1 vs RL/GFN) don't squash any curve against the x-axis.  The shaded band
	shows ±1 std across independent runs.

	Args:
		dataframes_dict: Dict mapping strategy name -> DataFrame.
		k:               Number of top sequences to average (default 10).
		oracle_col:      Column with oracle scores.
		iteration_col:   Column with iteration index.
		run_col:         Column identifying independent runs.

	Returns:
		(fig, axes) — the Figure and array of Axes.
	"""
	n = len(dataframes_dict)
	fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
	if n == 1:
		axes = [axes]

	colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

	for ax, (strategy_name, df), color in zip(axes, dataframes_dict.items(), colors):
		iterations, means, stds = _compute_topk_curve(df, k, oracle_col, iteration_col, run_col)

		ax.plot(iterations, means, marker='o', linewidth=2, markersize=5,
				color=color, label=strategy_name)
		ax.fill_between(iterations, means - stds, means + stds, alpha=0.15, color=color)

		ax.set_xlabel('Iterations (cumulative)', fontsize=11)
		ax.set_ylabel(f'Mean Top-{k} Oracle Score', fontsize=11)
		ax.set_title(strategy_name, fontsize=12)
		ax.set_xticks(iterations)
		ax.grid(alpha=0.3)

	fig.suptitle(f'Top-{k} Score vs. Number of Iterations', fontsize=13, y=1.02)
	plt.tight_layout()

	return fig, axes

