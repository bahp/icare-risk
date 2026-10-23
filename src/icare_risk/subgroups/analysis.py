from __future__ import annotations

from collections.abc import Mapping, Sequence
import time
import pandas as pd

from icare_risk.evaluation.metrics import compute_metrics
from icare_risk.evaluation.confidence import bootstrap_metrics
from .definitions import SubgroupDefinition
from .filtering import subgroup_mask


def evaluate_subgroups(
    df: pd.DataFrame,
    definitions: Sequence[SubgroupDefinition],
    *,
    target_column: str,
    prediction_columns: Sequence[str] | None = None,
    subset=None,
    threshold: float = 0.5,
    target_mapping: Mapping | None = None,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
    verbose: bool = False,
    progress_every: int = 1,
) -> pd.DataFrame:
    """Evaluate every prediction column within every configured subgroup.

    Set ``verbose=True`` to print progress and timing for each subgroup/model
    combination. Bootstrap evaluation can be computationally expensive because
    every combination performs its own resampling loop.
    """
    started = time.perf_counter()
    work = df.loc[subset].copy() if subset is not None else df.copy()
    prediction_columns = list(prediction_columns or [c for c in work.columns if c.endswith("_prob")])
    rows = []
    total = len(definitions) * len(prediction_columns)
    completed = 0

    if verbose:
        print(
            f"Starting subgroup evaluation: {len(work):,} rows, "
            f"{len(definitions)} subgroups, {len(prediction_columns)} models, "
            f"{total} evaluations, {n_bootstraps:,} bootstraps each.",
            flush=True,
        )

    for definition in definitions:
        subgroup_started = time.perf_counter()
        subgroup = work.loc[subgroup_mask(work, definition)]
        if verbose:
            print(
                f"Subgroup '{definition.name}': n={len(subgroup):,}",
                flush=True,
            )
        for prediction_column in prediction_columns:
            combination_started = time.perf_counter()
            point = compute_metrics(
                subgroup[target_column], subgroup[prediction_column],
                threshold=threshold, target_mapping=target_mapping,
            )
            interval = bootstrap_metrics(
                subgroup[target_column], subgroup[prediction_column],
                n_bootstraps=n_bootstraps, ci=ci, threshold=threshold,
                target_mapping=target_mapping, random_state=random_state,
            )
            rows.append({
                "subgroup": definition.name,
                "subgroup_column": definition.column,
                "subgroup_operator": definition.operator,
                "subgroup_value": definition.value,
                "model": prediction_column.removesuffix("_prob"),
                "prediction_column": prediction_column,
                **point,
                **interval,
            })
            completed += 1
            if verbose and (completed % progress_every == 0 or completed == total):
                elapsed = time.perf_counter() - started
                per_eval = elapsed / completed
                eta = per_eval * (total - completed)
                print(
                    f"  [{completed}/{total}] {definition.name} / "
                    f"{prediction_column}: {time.perf_counter() - combination_started:.1f}s; "
                    f"elapsed={elapsed:.1f}s, ETA={eta:.1f}s",
                    flush=True,
                )
        if verbose:
            print(
                f"Finished subgroup '{definition.name}' in "
                f"{time.perf_counter() - subgroup_started:.1f}s.",
                flush=True,
            )

    if verbose:
        print(f"Completed subgroup evaluation in {time.perf_counter() - started:.1f}s.", flush=True)
    return pd.DataFrame(rows)


def subgroup_sizes(
    df: pd.DataFrame,
    definitions: Sequence[SubgroupDefinition],
    *,
    target_column: str | None = None,
    target_mapping: Mapping | None = None,
) -> pd.DataFrame:
    """Summarise configured subgroup sizes and outcome counts."""
    rows = []
    for definition in definitions:
        subset = df.loc[subgroup_mask(df, definition)]
        row = {"subgroup": definition.name, "n": len(subset)}
        if target_column is not None:
            encoded = pd.Series(target_mapping or {})
            # Keep target counting consistent with compute_metrics without requiring predictions.
            from icare_risk.evaluation.metrics import encode_binary_target
            y = encode_binary_target(subset[target_column], target_mapping)
            row["n_events"] = int((y == 1).sum())
            row["n_non_events"] = int((y == 0).sum())
            row["n_unknown"] = int(y.isna().sum())
        rows.append(row)
    return pd.DataFrame(rows)
