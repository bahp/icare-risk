# Libraries
import importlib
import pandas as pd

def build_feature_matrix(dataset, config):
    """Iterates over the config, runs the functions, and joins the results."""
    feature_columns = []

    for feature_id, settings in config.items():
        print(f"Computing: {feature_id}...")

        # 1. Dynamically load the function based on the string in the config
        module_name = settings['module']
        function_name = settings['function']

        # If this was your real package, you'd use importlib:
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
        # func = globals()[function_name] # Using globals() here since everything is in one script

        # 2. Execute the function, passing the dataset and unpacking the kwargs
        feature_df = func(dataset=dataset, **settings['kwargs'])

        # 3. Set the index so pandas can easily join it with other features
        feature_df = feature_df.set_index(['SUBJECT', 'std_admission_time'])
        feature_columns.append(feature_df)

    # 4. Merge all individual flags into one wide dataframe
    final_matrix = pd.concat(feature_columns, axis=1)

    # Fill missing values with 0 (if a patient had no matching records, they get a 0 flag)
    return final_matrix.fillna(0).astype(int)