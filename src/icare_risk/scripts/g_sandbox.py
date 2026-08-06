import argparse
import importlib
import pandas as pd
import json
from pathlib import Path

from icare_risk.utils import load_yaml_config


def main():
    parser = argparse.ArgumentParser(description="Test a single YAML config entry.")
    parser.add_argument('entry_name', type=str, help="The name of the feature or score to test (e.g., 'hx_mi')")
    parser.add_argument('--config', type=str, default=None, help="Path to your local feature_config.yaml")
    parser.add_argument('--data', type=str, default=None, help="Path to a custom CSV (defaults to tests/cases.csv)")
    args = parser.parse_args()

    print(f"==================================================")
    print(f"🧪 Sandbox: Testing '{args.entry_name}'")
    print(f"==================================================\n")

    # 1. Load the configuration
    config = load_yaml_config(config_name="feature_config.yaml", user_path=args.config)

    # 2. Locate the requested entry in the YAML
    meta = None
    entry_type = None

    if args.entry_name in config.get('custom_features', {}):
        meta = config['custom_features'][args.entry_name]
        entry_type = "Feature"
    elif args.entry_name in config.get('custom_scores', {}):
        meta = config['custom_scores'][args.entry_name]
        entry_type = "Score"

    if not meta:
        print(f"❌ Error: '{args.entry_name}' not found in 'custom_features' or 'custom_scores'!")
        return

    # --- THE FIX: Pretty Print the Kwargs using JSON ---
    kwargs = meta.get('kwargs', {})
    pretty_kwargs = json.dumps(kwargs, indent=4)

    print(f"📌 Found {entry_type} configuration:")
    print(f"   Module:   {meta.get('module')}")
    print(f"   Function: {meta.get('function')}")
    print(f"   Kwargs:   \n{pretty_kwargs}\n")

    # --- THE FIX: Correct Path Resolution to the Project Root ---
    # Using Path.cwd() reliably points to the root where the Makefile is run from
    data_path = Path(args.data) if args.data else Path.cwd() / "tests" / "fixtures" / "cases.csv"

    if not data_path.exists():
        print(f"❌ Error: Could not find test data at {data_path}")
        print("   Generating a blank dummy dataframe instead...")
        df = pd.DataFrame({'patient_id': [1, 2, 3]})
    else:
        df = pd.read_csv(data_path)
        print(f"📄 Loaded test data: {len(df)} rows from {data_path.name}")

    # 4. Dynamically Load and Execute the Function
    module_name = meta.get('module')
    func_name = meta.get('function')

    try:
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)

        # Inject an empty mock context dictionary in case the function expects it
        kwargs['context_dfs'] = {}

        # Run it!
        result = func(df, **kwargs)
        df[args.entry_name] = result

    except Exception as e:
        print(f"\n❌ Execution Failed: {e}")
        return

    # 5. Display the Results neatly
    print("\n✅ Execution Successful!\n")

    used_cols = [v for k, v in kwargs.items() if isinstance(v, str) and v in df.columns]
    display_cols = ['patient_id'] + used_cols + [args.entry_name]
    display_cols = list(dict.fromkeys(display_cols))

    print("--- 📊 Input vs. Output ---")
    print(df[display_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()