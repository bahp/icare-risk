"""Phenotype registration — metadata-only decorator."""
from __future__ import annotations

import yaml
import importlib
import logging

from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Union

# -----------------------------------------------------------------------------
# Helper methods
# -----------------------------------------------------------------------------
def _validate_and_build_spec(name: str, spec_def: dict) -> Optional[PhenotypeSpec]:
    """Validates phenotype config and returns a PhenotypeSpec if valid, else None."""
    if not spec_def.get('enabled', True):
        return None

    mod_name = spec_def.get('module')
    func_name = spec_def.get('function')

    if not mod_name or not func_name:
        print(f"Skipping '{name}': missing module or function definition.")
        return None

    try:
        func = getattr(importlib.import_module(mod_name), func_name)
    except (ImportError, AttributeError) as e:
        print(f"Skipping '{name}': {type(e).__name__} - {e}")
        return None

    # Extract and pre-process codes once at startup if present in kwargs
    kwargs = spec_def.get('kwargs', {})
    if "codes" in kwargs:
        kwargs["codes"] = [str(c).upper() for c in kwargs["codes"]]
    #if "res195_codes" in kwargs:
    #    kwargs["res195_codes"] = [str(c).upper() for c in kwargs["res195_codes"]]

    return PhenotypeSpec(
        name=name,
        func=func,
        domains=spec_def.get('domains', []),
        category=spec_def.get('category', 'config_driven'),
        kwargs=kwargs # Exactly as YAML
    )


@dataclass(frozen=True)
class PhenotypeSpec:
    name: str
    func: Callable
    domains: List[str] = field(default_factory=list)
    category: str = "other"
    description: str = ""
    kwargs: Dict[str, Any] = field(default_factory=dict)


class PhenotypeRegistry:
    def __init__(self):
        self._specs: Dict[str, PhenotypeSpec] = {}

    def clear(self) -> None:
        self._specs.clear()

    def register(self, spec: PhenotypeSpec) -> None:
        #if spec.name in self._specs:
        #    raise ValueError(f"Phenotype '{spec.name}' is already registered.")
        self._specs[spec.name] = spec

    def get(self, name: str) -> PhenotypeSpec:
        return self._specs[name]

    def all(self, return_names: bool = False) -> List:
        """Returns all phenotype specs, or their names."""
        specs = list(self._specs.values())
        return [spec.name for spec in specs] if return_names else specs

    def select(self, names: Optional[List[str]] = None, return_names: bool = False) -> list:
        """Returns phenotype specs filtered by exact name, or their names."""
        if names is None:
            return self.all(return_names=return_names)
        specs = [self.get(n) for n in names]
        return [spec.name for spec in specs] if return_names else specs

    def select_by_prefix(self, prefix: str, return_names: bool = False) -> list:
        """Returns phenotype specs starting with a prefix, or names."""
        specs = [spec for name, spec in self._specs.items() if name.startswith(prefix)]
        return [spec.name for spec in specs] if return_names else specs

    def from_dict(self, config: dict) -> None:
        """Register phenotypes dynamically directly from a Python dictionary."""
        for name, spec_def in config.items():
            valid_spec = _validate_and_build_spec(name, spec_def)
            if valid_spec:
                self.register(valid_spec)

    def summary(self) -> "pd.DataFrame":
        """Returns a nicely formatted DataFrame of all registered phenotypes."""
        import pandas as pd

        rows = []
        for spec in self.all():  #
            rows.append({
                "Phenotype": spec.name,
                "Category": spec.category,
                "Domains": ", ".join(spec.domains) if spec.domains else "None",
                "Function": spec.func.__name__,
                "Parameters (kwargs)": str(spec.kwargs)
            })

        return pd.DataFrame(rows)

    def get_global_dependencies(self, names: Optional[List[str]] = None) -> tuple[Dict[str, set], Dict[str, set]]:
        """
        Scans registered phenotypes to build a unique set of required codes and columns per domain.
        Prevents 'keyword' extractors from pushing substring parameters into exact-match pushdowns.
        """
        from collections import defaultdict

        domain_codes = defaultdict(set)
        domain_cols = defaultdict(set)

        specs = self.select(names=names)

        for spec in specs:
            # 1. Handle Flat Configurations (Exact Codes)
            base_codes = spec.kwargs.get("codes", [])
            base_cols = spec.kwargs.get("columns", [])

            for domain in spec.domains:
                codes = spec.kwargs.get(f"{domain}_codes", base_codes)
                cols = spec.kwargs.get(f"{domain}_columns", base_cols)

                if codes:
                    if isinstance(codes, (list, tuple, set)):
                        domain_codes[domain].update(codes)
                    else:
                        domain_codes[domain].add(codes)
                if cols:
                    if isinstance(cols, (list, tuple, set)):
                        domain_cols[domain].update(cols)
                    else:
                        domain_cols[domain].add(cols)

            # 2. Handle Nested Components with Context Awareness
            components = spec.kwargs.get("components", [])
            for comp in components:
                comp_domain = comp.get("domain", spec.domains[0] if spec.domains else None)
                if not comp_domain:
                    continue

                extractor_type = comp.get("extractor_type", "rules")
                comp_col = comp.get("text_col") or comp.get("column") or "code"

                if extractor_type == "keyword":
                    # Keywords require Pandas .str.contains(), so they CANNOT be pushed to DuckDB's IN() clause.
                    # We just ensure the column is loaded into memory.
                    domain_cols[comp_domain].add(comp_col)
                else:
                    # For exact match extractors ('rules', 'expression', 'history')
                    comp_codes = comp.get("codes", [])
                    if comp_col == "code":
                        if comp_codes:
                            domain_codes[comp_domain].update(
                                comp_codes if isinstance(comp_codes, list) else [comp_codes]
                            )
                    else:
                        domain_cols[comp_domain].add(comp_col)

        return dict(domain_codes), dict(domain_cols)


DEFAULT_REGISTRY = PhenotypeRegistry()


def phenotype(
    name: str,
    domains: List[str],
    category: str = "other",
    description: str = "",
    registry: Optional[PhenotypeRegistry] = None,
):
    target_registry = registry or DEFAULT_REGISTRY

    def _decorator(func: Callable) -> Callable:
        spec = PhenotypeSpec(
            name=name,
            func=func,
            domains=list(domains),
            category=category,
            description=description or (func.__doc__ or "").strip(),
        )
        target_registry.register(spec)
        return func

    return _decorator


def register_from_yaml(yaml_path: str,
                       registry: PhenotypeRegistry = DEFAULT_REGISTRY):
    """Helper function to load phenotypes and runtime kwargs straight from a YAML file."""
    load_phenotypes_from_yaml(yaml_path, registry)
    #import yaml
    #with open(yaml_path, 'r') as f:
    #    config = yaml.safe_load(f)
    #registry.from_dict(config)


def load_phenotypes_from_yaml(
        paths: Union[str, Path, List[Union[str, Path]]],
        registry: PhenotypeRegistry = DEFAULT_REGISTRY,
        reset: bool = True
) -> PhenotypeRegistry:
    """Helper function to load phenotypes from a single YAML file, a list of files,
    or an entire directory containing YAML files.
    """
    if registry is None:
        registry = PhenotypeRegistry()

    if reset:
        registry.clear()

    resolved_paths: List[Path] = []

    # Normalize inputs into a flat list of Path objects
    inputs = [paths] if isinstance(paths, (str, Path)) else paths

    for item in inputs:
        p = Path(item)
        if p.is_dir():
            resolved_paths.extend(sorted(p.glob("*.yaml")))
            resolved_paths.extend(sorted(p.glob("*.yml")))
        elif p.is_file():
            resolved_paths.append(p)
        else:
            raise FileNotFoundError(f"Path not found or invalid: {p}")

    # Load each configuration file sequentially into the registry
    for yaml_path in resolved_paths:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        if config:
            registry.from_dict(config)

    # Return
    return registry