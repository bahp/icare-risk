"""clinphen -- modular, leakage-safe clinical phenotype & feature matrix engine."""
from .config.schema import DEFAULT_SCHEMA, EpisodeTableSchema, SchemaConfig, TableSchema
from .engine.runner import FeatureMatrixBuilder
from .io.cohort_cache import CohortDataCache
from .io.loaders import DuckDBSource
from .registry.registry import DEFAULT_REGISTRY, PhenotypeRegistry, PhenotypeSpec, phenotype
from .temporal.context import EpisodeContext

from .phenotypes import windowed as _windowed  # noqa: F401

__all__ = [
    "SchemaConfig",
    "TableSchema",
    "EpisodeTableSchema",
    "DEFAULT_SCHEMA",
    "DuckDBSource",
    "CohortDataCache",
    "EpisodeContext",
    "phenotype",
    "PhenotypeRegistry",
    "PhenotypeSpec",
    "DEFAULT_REGISTRY",
    "FeatureMatrixBuilder",
]
