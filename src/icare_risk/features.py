# src/features.py
import yaml
import importlib
import pandas as pd
import numpy as np
from pathlib import Path


class FeaturePipeline:
    def __init__(self, config_dict=None,
                       data_config=None,
                       context_dfs=None,
                       targets=None):
        self.config = config_dict or {}
        self.data_config = data_config or {}
        self.context_dfs = context_dfs or {}
        self.targets = targets or []
        self.patient_col = self.data_config \
            .get('pipeline_settings', {}) \
            .get('patient_col')
        self.encounter_col = self.data_config \
            .get('pipeline_settings', {}) \
            .get('encounter_col')

    def process(self, df_static, df_ts):
        """Main orchestration method.
        # df_static is your episodes table
        # df_ts is your pivoted vitals/labs
        """
        #df = df_ts.reset_index()
        df = df_ts \
            .sort_values([self.patient_col, 'date']) \
            .set_index([self.patient_col, 'date'])

        print("Processing base features...")

        # Processing
        df = self._process_base_features(df)
        df = self._compute_custom_features(df)
        df = df.reset_index()
        df = pd.merge(df, df_static, on=[self.patient_col], how='left')
        df = self._compute_expressions(df)
        df = self._compute_custom_scores(df)
        df = df.reset_index()

        # Return
        return df

    def _process_base_features(self, df):
        """Handles missing indicators, imputation, deltas, and rolling statistics."""
        new_columns = []

        for col, rules in self.config.get('base_features', {}).items():
            if col not in df.columns:
                continue

            # A. Missing Indicators
            if rules.get('missing_indicator'):
                new_columns.append(df[col].isna().astype(int).rename(f"{col}_is_missing"))

            # B. Imputation (grouped by patient to prevent cross-contamination)
            grp = df.groupby(level=self.patient_col)[col]

            if rules.get('impute') == 'ffill':
                df[col] = grp.ffill()
            elif rules.get('impute') == 'constant':
                df[col] = df[col].fillna(rules.get('fill_value', 0))

            # Fallback for leading NaNs
            df[col] = df[col].fillna(df[col].median())

            # C. Deltas
            if rules.get('delta'):
                new_columns.append(grp.diff().fillna(0).rename(f"{col}_delta"))

            # D. Rolling Time-Windows
            if 'rolling' in rules:
                # For time-aware rolling, date must be the only index
                temp_df = df[[col]].reset_index(level=self.patient_col)

                for window in rules['rolling'].get('windows', []):
                    rolled = temp_df.groupby(self.patient_col)[col].rolling(window).agg(rules['rolling']['aggs'])
                    rolled.columns = [f"{col}_{window}_{agg}" for agg in rolled.columns]
                    new_columns.append(rolled)

        # Concatenate all newly generated features at once (Highly optimized memory usage)
        if new_columns:
            df = pd.concat([df] + new_columns, axis=1)

        return df

    def _compute_custom_features(self, df):
        custom_feats = self.config.get('custom_features', {})
        df_static = getattr(self, 'context_dfs', {}).get('episodes')

        # 1. Load Centralized Settings
        pid_col = self.patient_col
        enc_col = self.encounter_col
        adm_col = 'ADMISSION_DATE'
        ts_date_col = 'date'

        # Pre-import functions
        active_features = {}
        for feat_name, meta in custom_feats.items():
            if self.targets and feat_name not in self.targets: continue
            module = importlib.import_module(meta.get('module', 'icare_risk.phenotypes'))
            active_features[feat_name] = (getattr(module, meta.get('function')), meta.get('kwargs', {}))

        if not active_features:
            return df

        results = []

        print(df.index)
        print(df.columns)

        # 2. Group using generic central columns
        for (pid, enc_id), stay_df in df.groupby([pid_col, enc_col]):

            print(pid)
            print(enc_id)

            stay_info = df_static[(df_static[pid_col] == pid) & (df_static[enc_col] == enc_id)]
            if stay_info.empty:
                results.append(stay_df)
                continue

            # Extract admission date and anchor time dynamically
            adm_date = pd.to_datetime(stay_info[adm_col].iloc[0])
            anchor_time = stay_df.index.get_level_values(ts_date_col).max() if ts_date_col in stay_df.index.names else \
            stay_df[ts_date_col].max()

            past_context, current_context = {}, {}

            # 3. Generic Context Iteration
            for ctx_name, ctx_df in getattr(self, 'context_dfs', {}).items():
                if ctx_df is None or ctx_df.empty or ctx_name == 'episodes':
                    continue

                schema = schemas.get(ctx_name, {})
                ctx_pid_col = schema.get('id_col', pid_col)
                date_col = schema.get('date_col')

                if ctx_pid_col in ctx_df.columns:
                    p_data = ctx_df[ctx_df[ctx_pid_col] == pid].copy()
                else:
                    p_data = ctx_df.copy()

                if date_col and date_col in p_data.columns:
                    p_data[date_col] = pd.to_datetime(p_data[date_col])
                    past_context[ctx_name] = p_data[p_data[date_col] < adm_date]
                    current_context[ctx_name] = p_data[
                        (p_data[date_col] >= adm_date) & (p_data[date_col] <= anchor_time)]
                else:
                    past_context[ctx_name] = p_data
                    current_context[ctx_name] = p_data

            # 4. Execute Features
            stay_df_copy = stay_df.copy()
            for feat_name, (func, kwargs) in active_features.items():
                kwargs['past_context'] = past_context
                kwargs['current_context'] = current_context
                stay_df_copy[feat_name] = func(stay_df_copy, **kwargs)

            results.append(stay_df_copy)

        return pd.concat(results).sort_index() if results else df


    def _compute_custom_features_v1(self, df):
        """Executes the custom_features block from YAML (Phenotypes)."""
        custom_feats = self.config.get('custom_features', {})
        for feat_name, meta in custom_feats.items():

            if self.targets and feat_name not in self.targets:
                continue

            module_name = meta.get('module', 'icare_risk.phenotypes')
            func_name = meta.get('function')
            kwargs = meta.get('kwargs', {})

            # Pass context_dfs if you are using the 'Lazy Lookup' approach
            kwargs['context_dfs'] = getattr(self, 'context_dfs', {})

            try:
                module = importlib.import_module(module_name)
                func = getattr(module, func_name)
                print(f"  🔍 Computing phenotype: {feat_name}...")
                result = func(df, **kwargs)
                df[feat_name] = pd.Series(result, index=df.index)
                print(f"     ✅ {feat_name} added. Unique values: {df[feat_name].unique()}")
            except Exception as e:
                msg = f"⚠️ Warning: Failed to compute phenotype '{feat_name}'. Error: {e}"
                if self.config.get('strict_mode', True):
                    raise RuntimeError(f"❌ {msg}") from e
                else:
                    print(f"⚠️ Warning: {msg}")
        return df


    def _compute_expressions(self, df):
        """Dynamically evaluates string expressions to create new features/scores.

        . note: The numexpr is fast, but does not handle True + True. So we can
                tell Pandas to use the standard Python engine so it can safely
                add boolean flags (True + True = 2) without numexpr complaining!
                df[feature_name] = df.eval(expr, engine='python')

        Parameters
        ----------

        Returns
        -------
        """
        expressions = self.config.get('computed_features', {})
        for feature_name, expr in expressions.items():
            if self.targets and feature_name not in self.targets:
                continue
            try:
                df[feature_name] = df.eval(expr, engine='python')
            except Exception as e:
                print(f"Warning: Failed to compute '{feature_name}'. Error: {e}")

        return df


    def _compute_custom_scores(self, df):
        """Dynamically imports and executes external Python scoring functions."""
        import importlib

        custom_scores = self.config.get('custom_scores', {})

        for score_name, meta in custom_scores.items():
            if self.targets and score_name not in self.targets:
                continue

            module_name = meta.get('module', 'scores')
            func_name = meta.get('function')
            kwargs = meta.get('kwargs', {})

            try:
                # Dynamically load the module and function
                module = importlib.import_module(module_name)
                custom_func = getattr(module, func_name)

                # --- 1. PRE-COMPUTATION LOG ---
                print(f"  🔍 Computing score: {score_name}...")

                # Execute the function
                kwargs['context_dfs'] = getattr(self, 'context_dfs', {})
                df[score_name] = custom_func(df, **kwargs)

                # --- 2. SUCCESS LOG WITH UNIQUE VALUES ---
                # To keep the terminal clean, we format the array slightly if there are many unique values
                unique_vals = df[score_name].unique()
                if len(unique_vals) > 10:
                    val_str = f"[{len(unique_vals)} unique values]"
                else:
                    val_str = f"{unique_vals}"

                print(f"     ✅ {score_name} added. Unique values: {val_str}")

            except ModuleNotFoundError:
                print(f"❌ Error computing '{score_name}': Cannot find module '{module_name}'.")
                if not module_name.startswith('icare_risk.'):
                    print(
                        f"   💡 Hint: If your file is in the 'src.icare_risk' folder, update your YAML to use `module: 'icare_risk.{module_name}'`")

            except AttributeError:
                print(
                    f"❌ Error computing '{score_name}': Function '{func_name}' does not exist inside '{module_name}'.")

            #except KeyError as e:
            #    print(f"❌ Error computing '{score_name}': Missing required column {e}.")
            #    print(
            #        f"   💡 Hint: Check if this column is generated in Step 1 or defined in your 'custom_features' YAML block.")

            except Exception as e:
                msg = f"⚠️ Warning: Failed to compute custom score '{score_name}'. Error: {e}"
                if self.config.get('strict_mode', True):
                    raise RuntimeError(f"❌ {msg}") from e
                else:
                    print(f"⚠️ Warning: {msg}")

        return df

if __name__ == '__main__':

    # 1. Setup paths
    latest_dir = sorted([d for d in Path('../../data/synthetic').iterdir() if d.is_dir()])[-1]

    # 2. Load data
    df_static = pd.read_csv(latest_dir / 'df_static.csv')
    df_ts_ms = pd.read_csv(latest_dir / 'df_ts_missing.csv', parse_dates=['date'])

    # 3. Run Pipeline
    pipeline = FeaturePipeline()
    final_features = pipeline.process(df_static, df_ts_ms)

    # 4. Show results
    print("\n--- Pipeline Complete! ---")
    print(f"Final Shape: {final_features.shape}")

    preview_cols = [self.patient_col, 'date', 'HR', 'SBP', 'Shock_Index', 'qSOFA_score']
    available = [c for c in preview_cols if c in final_features.columns]
    print("\nSample of Computed Scores & Ratios:")
    print(final_features[available].dropna().head())