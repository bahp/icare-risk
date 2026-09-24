import cProfile
import pstats
import io
import duckdb

from icare_risk.clinphen import DuckDBSource
from icare_risk.clinphen.registry.registry import DEFAULT_REGISTRY
from icare_risk.clinphen.registry.registry import register_from_yaml
from icare_risk.clinphen.config.schema import load_schema_from_yaml
from icare_risk.clinphen.engine.runner import FeatureMatrixBuilder


def main():
    # Define paths
    schema_cfg = 'src/icare_risk/config/icare/schema.yaml'
    phenotype_cfg = 'src/icare_risk/config/icare/phenotypes.yaml'

    # Load schema definition & create connection source
    schema = load_schema_from_yaml(schema_cfg)
    source = DuckDBSource(connection=duckdb.connect())

    # Load the phenotypes definitions
    DEFAULT_REGISTRY._specs.clear()
    register_from_yaml(phenotype_cfg, DEFAULT_REGISTRY)

    charlson = DEFAULT_REGISTRY.select_by_prefix('charlson', return_names=True)

    # Create matrix builder
    builder = FeatureMatrixBuilder(
        schema=schema, source=source,
        registry=DEFAULT_REGISTRY,
        phenotypes=charlson
    )

    print("Starting feature matrix build profiling...")

    # Initialize and run profiler specifically on the build step
    profiler = cProfile.Profile()
    profiler.enable()

    feature_matrix = builder.build(raise_on_error=True)

    profiler.disable()

    # Process and print stats sorted by cumulative time
    s = io.StringIO()
    sortby = 'cumtime'
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.print_stats(25)  # Top 25 most expensive functions/calls
    print(s.getvalue())

    # Optional: Save profile results for visualization (e.g., snakeviz)
    profiler.dump_stats("feature_matrix_profile.prof")
    print(
        "Profile results saved to 'feature_matrix_profile.prof'. Visualize with: snakeviz feature_matrix_profile.prof")


if __name__ == '__main__':
    main()