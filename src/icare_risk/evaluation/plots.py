"""Reusable plots for model performance, calibration, and temporal analyses.

The functions in this module return Matplotlib ``Figure`` objects. Notebooks
choose which metrics and models to display and decide where figures are saved.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
from math import ceil

import matplotlib.pyplot as plt
import pandas as pd


_DEFAULT_COLORS = list(plt.get_cmap("tab10").colors)


def _validate_metrics(results: pd.DataFrame, metrics: Sequence[str]) -> list[str]:
    missing = [metric for metric in metrics if metric not in results.columns]
    if missing:
        raise KeyError(f"Metrics not found in results: {missing}")
    return list(metrics)


def _filtered_results(
    results: pd.DataFrame,
    model_names: Sequence[str] | None = None,
    model_column: str = "model",
) -> pd.DataFrame:
    if results.empty:
        raise ValueError("Cannot plot an empty results table.")
    work = results.copy()
    if model_names is not None:
        if model_column not in work.columns:
            raise KeyError(f"Model column '{model_column}' not found.")
        work = work[work[model_column].isin(model_names)].copy()
    if work.empty:
        raise ValueError("No rows remain after filtering model_names.")
    return work


def _period_values(work: pd.DataFrame, period_column: str) -> pd.Series:
    if period_column not in work.columns:
        raise KeyError(f"Period column '{period_column}' not found.")
    values = work[period_column]
    if pd.api.types.is_datetime64_any_dtype(values):
        return values
    parsed = pd.to_datetime(values, errors="coerce")
    return parsed if parsed.notna().any() else values.astype(str)


def plot_temporal_metrics(
    results: pd.DataFrame,
    *,
    metrics: Sequence[str],
    model_names: Sequence[str] | None = None,
    period_column: str = "period",
    model_column: str = "model",
    n_columns: int = 2,
    sharex: bool = True,
    sharey: bool = False,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    y_limits: dict[str, tuple[float, float]] | None = None,
    show_points: bool = True,
    show_sample_size: bool = False,
) -> plt.Figure:
    """Plot one or more metrics over calendar periods using small multiples.

    Parameters
    ----------
    results:
        Long-format temporal results with one row per model and period.
    metrics:
        Metric column names to plot, for example ``["auroc", "auprc"]``.
    model_names:
        Optional model names to retain. By default all models are shown.
    period_column, model_column:
        Names of the calendar-period and model columns.
    n_columns:
        Number of subplot columns.
    sharex, sharey:
        Whether subplots share their x or y axis.
    figsize:
        Optional overall figure size.
    title:
        Optional figure title.
    y_limits:
        Optional metric-specific axis limits.
    show_points:
        Draw markers at each period.
    show_sample_size:
        Annotate points with ``n_admissions`` when that column exists.
    """
    work = _filtered_results(results, model_names, model_column)
    metrics = _validate_metrics(work, metrics)
    x_values = _period_values(work, period_column)
    work = work.assign(_plot_period=x_values)

    n_columns = max(1, min(n_columns, len(metrics)))
    n_rows = ceil(len(metrics) / n_columns)
    if figsize is None:
        figsize = (7 * n_columns, 4.5 * n_rows)
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=figsize, sharex=sharex, sharey=sharey,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    models = list(work[model_column].dropna().unique())
    colors = {model: _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i, model in enumerate(models)}

    for ax, metric in zip(axes_flat, metrics):
        for model in models:
            group = work[work[model_column].eq(model)].sort_values("_plot_period")
            kwargs = {"color": colors[model], "label": str(model), "linewidth": 2}
            if show_points:
                kwargs.update(marker="o", markersize=5)
            ax.plot(group["_plot_period"], group[metric], **kwargs)
            lower = f"{metric}_ci_lower"
            upper = f"{metric}_ci_upper"
            if lower in group.columns and upper in group.columns:
                ax.fill_between(
                    group["_plot_period"], group[lower], group[upper],
                    color=colors[model], alpha=0.12, linewidth=0,
                )
            if show_sample_size and "n_admissions" in group.columns:
                for _, row in group.iterrows():
                    if pd.notna(row[metric]) and pd.notna(row["n_admissions"]):
                        ax.annotate(
                            f"n={int(row['n_admissions'])}",
                            (row["_plot_period"], row[metric]),
                            xytext=(0, 6), textcoords="offset points",
                            fontsize=7, alpha=0.7, ha="center",
                        )
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel(metric.replace("_", " ").title())
        if y_limits and metric in y_limits:
            ax.set_ylim(*y_limits[metric])
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", rotation=45)

    for ax in axes_flat[len(metrics):]:
        ax.set_visible(False)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)),
                   bbox_to_anchor=(0.5, 1.02), frameon=False)
    if title:
        fig.suptitle(title, y=1.08 if handles else 1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_temporal_metric(
    results: pd.DataFrame,
    *,
    metric: str,
    **kwargs,
) -> plt.Figure:
    """Convenience wrapper for plotting one temporal metric."""
    return plot_temporal_metrics(results, metrics=[metric], **kwargs)


def plot_calibration_metrics(
    results: pd.DataFrame,
    *,
    model_names: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    **kwargs,
) -> plt.Figure:
    """Plot calibration metrics over time as small multiples."""
    if metrics is None:
        preferred = ["brier_score", "calibration_intercept", "calibration_slope", "observed_expected_ratio"]
        metrics = [metric for metric in preferred if metric in results.columns]
    if not metrics:
        raise ValueError("No calibration metric columns were found.")
    return plot_temporal_metrics(
        results, metrics=metrics, model_names=model_names,
        y_limits=kwargs.pop("y_limits", None), **kwargs,
    )


def plot_threshold_metrics(
    results: pd.DataFrame,
    *,
    metrics: Sequence[str] = ("sensitivity", "specificity", "ppv", "npv"),
    model_names: Sequence[str] | None = None,
    threshold_label: str | None = None,
    **kwargs,
) -> plt.Figure:
    """Plot fixed-threshold classification metrics over time."""
    title = kwargs.pop("title", None)
    if threshold_label:
        suffix = f"Fixed threshold: {threshold_label}"
        title = f"{title} ({suffix})" if title else suffix
    return plot_temporal_metrics(
        results, metrics=metrics, model_names=model_names,
        y_limits=kwargs.pop("y_limits", {metric: (0, 1) for metric in metrics}),
        title=title, **kwargs,
    )


def plot_calibration_curve(
    table: pd.DataFrame,
    *,
    model_column: str = "model",
    mean_predicted_column: str = "mean_predicted",
    observed_column: str = "observed_rate",
    model_names: Sequence[str] | None = None,
    figsize: tuple[float, float] = (7, 6),
    title: str = "Calibration plot",
) -> plt.Figure:
    """Plot observed event rate against mean predicted risk."""
    required = {model_column, mean_predicted_column, observed_column}
    missing = required.difference(table.columns)
    if missing:
        raise KeyError(f"Calibration columns not found: {sorted(missing)}")
    work = table if model_names is None else table[table[model_column].isin(model_names)]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], "--", color="black", alpha=0.6, label="Perfect calibration")
    for i, (model, group) in enumerate(work.groupby(model_column)):
        group = group.sort_values(mean_predicted_column)
        ax.plot(group[mean_predicted_column], group[observed_column], marker="o",
                linewidth=2, color=_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)], label=str(model))
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed ESBL event rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig




def plot_temporal_event_rate(
    results: pd.DataFrame,
    *,
    period_column: str = "period",
    event_rate_column: str = "event_rate",
    admissions_column: str | None = None,
    events_column: str = "n_events",
    title: str = "ESBL event rate over calendar time",
    figsize: tuple[float, float] = (10, 5),
    y_limits: tuple[float, float] | None = (0, 1),
) -> plt.Figure:
    """Plot outcome prevalence by calendar period.

    ``temporal_split_summary`` names the denominator ``n_resolved`` because
    rows with unresolved outcomes are excluded from the event-rate
    denominator. If ``admissions_column`` is not supplied, this function
    uses ``n_resolved`` and falls back to ``n_rows`` or ``n_admissions``.

    Parameters
    ----------
    results:
        Temporal cohort summary, usually returned by
        :func:`temporal_split_summary`.
    period_column:
        Calendar-period column.
    event_rate_column:
        Column containing the event proportion.
    admissions_column:
        Column used for the denominator annotation. Defaults to the first
        available column among ``n_resolved``, ``n_rows``, and
        ``n_admissions``.
    events_column:
        Column containing the number of events.
    title, figsize, y_limits:
        Figure customisation options.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing one event-rate point per calendar period.
    """
    required = {period_column, event_rate_column, events_column}
    missing = required.difference(results.columns)
    if missing:
        raise KeyError(f"Missing event-rate columns: {sorted(missing)}")

    if admissions_column is None:
        admissions_column = next(
            (column for column in ("n_resolved", "n_rows", "n_admissions")
             if column in results.columns),
            None,
        )
    if admissions_column is None:
        raise KeyError(
            "Could not find a denominator column. Expected one of "
            "'n_resolved', 'n_rows', or 'n_admissions'."
        )

    work = results.copy()
    work["_plot_period"] = _period_values(work, period_column)
    work = work.sort_values("_plot_period")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        work["_plot_period"],
        pd.to_numeric(work[event_rate_column], errors="coerce"),
        marker="o",
        linewidth=2,
        color=_DEFAULT_COLORS[0],
        label="ESBL event rate",
    )
    for _, row in work.iterrows():
        rate = pd.to_numeric(pd.Series([row[event_rate_column]]), errors="coerce").iloc[0]
        denominator = pd.to_numeric(pd.Series([row[admissions_column]]), errors="coerce").iloc[0]
        events = pd.to_numeric(pd.Series([row[events_column]]), errors="coerce").iloc[0]
        if pd.notna(rate) and pd.notna(denominator) and pd.notna(events):
            ax.annotate(
                f"n={int(denominator):,}\nevents={int(events):,}",
                (row["_plot_period"], rate),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                alpha=0.75,
            )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Calendar period")
    ax.set_ylabel("ESBL event rate")
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(alpha=0.25)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_model_metrics(
    results: pd.DataFrame,
    *,
    metrics: Sequence[str],
    model_column: str = "model",
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    y_limits: dict[str, tuple[float, float]] | None = None,
) -> plt.Figure:
    """Plot model-level metrics as comparable small-multiple bar charts.

    Parameters
    ----------
    results:
        One row per model containing the requested metric columns.
    metrics:
        Metric columns to display.
    model_column:
        Column identifying models.
    figsize, title, y_limits:
        Optional figure and axis customisation.
    """
    work = _filtered_results(results, model_column=model_column)
    metrics = _validate_metrics(work, metrics)
    n_columns = min(2, len(metrics))
    n_rows = ceil(len(metrics) / n_columns)
    fig, axes = plt.subplots(
        n_rows, n_columns,
        figsize=figsize or (7 * n_columns, 4.5 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    labels = work[model_column].astype(str).tolist()
    for ax, metric in zip(axes_flat, metrics):
        values = pd.to_numeric(work[metric], errors="coerce")
        ax.bar(labels, values, color=_DEFAULT_COLORS[:len(labels)])
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel(metric.replace("_", " ").title())
        if y_limits and metric in y_limits:
            ax.set_ylim(*y_limits[metric])
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes_flat[len(metrics):]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig



def plot_subgroup_performance_dashboard(
    results: pd.DataFrame,
    *,
    metrics: Sequence[str] = ("auroc", "auprc", "sensitivity", "specificity"),
    model_names: Sequence[str] | None = None,
    subgroup_order: Sequence[str] | None = None,
    model_column: str = "model",
    subgroup_column: str = "subgroup",
    n_columns: int = 2,
    figsize: tuple[float, float] | None = None,
    title: str = "Model performance across subgroups",
    value_limits: dict[str, tuple[float, float]] | None = None,
    show_confidence_intervals: bool = True,
    minimum_n: int | None = None,
    minimum_events: int | None = None,
) -> plt.Figure:
    """Create a publication-style dashboard of subgroup performance.

    Each panel is a heatmap for one metric, with subgroups on the y-axis and
    models on the x-axis. Cell annotations show the point estimate and, when
    available, its confidence interval. Subgroup size and event counts are
    included in the y-axis labels so unstable estimates remain visible.

    Parameters
    ----------
    results:
        Long-format subgroup results with one row per subgroup/model pair.
    metrics:
        Metrics to display. Typical choices are ``["auroc", "auprc",
        "sensitivity", "specificity"]`` or ``["brier_score", "ppv", "npv"]``.
    model_names:
        Optional model order/filter. By default models are ordered as found.
    subgroup_order:
        Optional subgroup order. By default subgroups are ordered as found.
    model_column, subgroup_column:
        Column names identifying models and subgroups.
    n_columns:
        Number of metric panels per row.
    figsize:
        Overall figure size. A size is inferred when omitted.
    title:
        Figure title.
    value_limits:
        Metric-specific colour limits. Defaults to ``(0, 1)`` for each metric.
    show_confidence_intervals:
        Annotate cells with confidence intervals when matching columns exist.
    minimum_n, minimum_events:
        Optional minimum sample/event counts. Cells failing the threshold are
        shown as unavailable rather than interpreted as reliable estimates.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing one heatmap panel per requested metric.
    """
    if results.empty:
        raise ValueError("Cannot plot an empty subgroup results table.")
    metrics = _validate_metrics(results, metrics)
    required = {model_column, subgroup_column}
    missing = required.difference(results.columns)
    if missing:
        raise KeyError(f"Subgroup columns not found: {sorted(missing)}")

    work = results.copy()
    if model_names is None:
        models = list(work[model_column].dropna().astype(str).unique())
    else:
        models = [str(model) for model in model_names]
        work = work[work[model_column].astype(str).isin(models)].copy()
    if not models:
        raise ValueError("No models remain after filtering.")

    if subgroup_order is None:
        subgroups = list(work[subgroup_column].dropna().astype(str).unique())
    else:
        subgroups = [str(name) for name in subgroup_order]
        work = work[work[subgroup_column].astype(str).isin(subgroups)].copy()
    if not subgroups:
        raise ValueError("No subgroups remain after filtering.")

    work[model_column] = work[model_column].astype(str)
    work[subgroup_column] = work[subgroup_column].astype(str)
    counts = work.drop_duplicates([subgroup_column]).set_index(subgroup_column)

    labels = []
    for subgroup in subgroups:
        if subgroup in counts.index:
            row = counts.loc[subgroup]
            n = row.get("n", row.get("n_resolved", None))
            events = row.get("n_events", None)
            suffix = []
            if pd.notna(n):
                suffix.append(f"n={int(n):,}")
            if pd.notna(events):
                suffix.append(f"events={int(events):,}")
            labels.append(f"{subgroup} ({'; '.join(suffix)})" if suffix else subgroup)
        else:
            labels.append(subgroup)

    n_columns = max(1, min(n_columns, len(metrics)))
    n_rows = ceil(len(metrics) / n_columns)
    if figsize is None:
        figsize = (7 * n_columns, max(4.5, 0.58 * len(subgroups) + 1.8) * n_rows)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for ax, metric in zip(axes_flat, metrics):
        matrix = (
            work.pivot_table(index=subgroup_column, columns=model_column, values=metric, aggfunc="first")
            .reindex(index=subgroups, columns=models)
        )
        values = matrix.to_numpy(dtype=float)
        limits = (value_limits or {}).get(metric, (0.0, 1.0))
        image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=limits[0], vmax=limits[1])

        for i in range(len(subgroups)):
            for j in range(len(models)):
                value = values[i, j]
                valid = pd.notna(value)
                if valid and minimum_n is not None and subgroups[i] in counts.index:
                    n_value = counts.loc[subgroups[i]].get("n", counts.loc[subgroups[i]].get("n_resolved", None))
                    valid = pd.notna(n_value) and n_value >= minimum_n
                if valid and minimum_events is not None and subgroups[i] in counts.index:
                    event_value = counts.loc[subgroups[i]].get("n_events", None)
                    valid = pd.notna(event_value) and event_value >= minimum_events

                if not valid:
                    text = "—"
                else:
                    text = f"{value:.2f}"
                    lower = f"{metric}_ci_lower"
                    upper = f"{metric}_ci_upper"
                    if show_confidence_intervals and lower in work.columns and upper in work.columns:
                        cell = work[(work[subgroup_column] == subgroups[i]) & (work[model_column] == models[j])]
                        if not cell.empty and pd.notna(cell.iloc[0][lower]) and pd.notna(cell.iloc[0][upper]):
                            text += f" [{cell.iloc[0][lower]:.2f}, {cell.iloc[0][upper]:.2f}]"
                ax.text(j, i, text, ha="center", va="center", fontsize=7,
                        color="white" if valid and pd.notna(value) and value >= (limits[0] + limits[1]) / 2 else "black")

        # Major ticks are the only labelled positions: one per model column
        # and one per subgroup row. Minor ticks mark only internal cell borders.
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_yticks(range(len(subgroups)))
        ax.set_yticklabels(labels)
        ax.set_title(metric.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel("Model")
        ax.set_ylabel("Subgroup")

        # Disable any inherited/style-level grid first. Then draw only the
        # internal boundaries between heatmap cells; this prevents duplicate
        # lines through labelled ticks.
        ax.grid(False, which="both")
        if len(models) > 1:
            ax.set_xticks([x - 0.5 for x in range(1, len(models))], minor=True)
        if len(subgroups) > 1:
            ax.set_yticks([y - 0.5 for y in range(1, len(subgroups))], minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=metric.replace("_", " ").title())

    for ax in axes_flat[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_subgroup_performance_forest(
    results: pd.DataFrame,
    *,
    metric: str = "auroc",
    model_name: str | None = None,
    model_column: str = "model",
    subgroup_column: str = "subgroup",
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    x_limits: tuple[float, float] = (0, 1),
) -> plt.Figure:
    """Plot one model's subgroup metric as a confidence-interval forest plot."""
    required = {model_column, subgroup_column, metric}
    missing = required.difference(results.columns)
    if missing:
        raise KeyError(f"Subgroup columns not found: {sorted(missing)}")
    work = results.copy()
    if model_name is not None:
        work = work[work[model_column].eq(model_name)].copy()
    if work.empty:
        raise ValueError("No subgroup rows remain for the requested model.")
    work = work.sort_values(metric).reset_index(drop=True)
    height = max(4.5, 0.42 * len(work) + 1.5)
    fig, ax = plt.subplots(figsize=figsize or (8, height))
    y = range(len(work))
    lower = work.get(f"{metric}_ci_lower", work[metric])
    upper = work.get(f"{metric}_ci_upper", work[metric])
    ax.errorbar(work[metric], y, xerr=[work[metric] - lower, upper - work[metric]],
                fmt="o", color="#1f4e79", ecolor="#7f8c8d", capsize=3, linewidth=1.5)
    ax.set_yticks(list(y), work[subgroup_column].astype(str))
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_xlim(*x_limits)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title(title or f"{metric.replace('_', ' ').title()} across subgroups", fontweight="bold")
    fig.tight_layout()
    return fig

def save_figure(
    figure: plt.Figure,
    path: str | Path,
    *,
    dpi: int = 300,
    close: bool = False,
    **savefig_kwargs,
) -> Path:
    """Save an existing Matplotlib figure without redrawing it.

    Parameters
    ----------
    figure:
        Figure returned by one of the plotting functions in this module.
    path:
        Output path, including the desired image extension such as ``.png`` or
        ``.pdf``. Parent directories are created automatically.
    dpi:
        Resolution passed to :meth:`matplotlib.figure.Figure.savefig`.
    close:
        If ``True``, close the figure after saving. Keep ``False`` when the
        figure is still being displayed or may be saved in another format.
    **savefig_kwargs:
        Additional keyword arguments passed to ``figure.savefig``.

    Returns
    -------
    pathlib.Path
        The saved output path.
    """
    from pathlib import Path

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", **savefig_kwargs)
    if close:
        plt.close(figure)
    return output
