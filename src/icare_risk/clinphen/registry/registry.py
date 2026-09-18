"""Phenotype registration — metadata-only decorator."""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any

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

    return PhenotypeSpec(
        name=name,
        func=func,
        domains=spec_def.get('domains', []),
        category=spec_def.get('category', 'config_driven'),
        kwargs=spec_def.get('kwargs', {})
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
    import yaml
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    registry.from_dict(config)