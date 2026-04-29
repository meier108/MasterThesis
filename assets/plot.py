import numpy as np
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

