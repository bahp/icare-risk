from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SubgroupDefinition:
    """A subgroup defined by one condition or several combined conditions."""

    name: str
    column: str | None = None
    operator: str | None = None
    value: Any = None
    description: str | None = None
    conditions: tuple[Mapping[str, Any], ...] = ()
    logic: str = "all"

    @property
    def is_compound(self) -> bool:
        return bool(self.conditions)


def load_subgroup_definitions(config: Mapping) -> list[SubgroupDefinition]:
    """Load simple and compound subgroup definitions from YAML/configuration.

    Simple definition::

        older:
          column: AGE_AT_ADMISSION
          operator: ">="
          value: 65

    Compound definition::

        older_female_diabetes:
          logic: all
          conditions:
            - {column: AGE_AT_ADMISSION, operator: ">=", value: 65}
            - {column: GENDER_DESC, operator: "==", value: Female}
            - {column: charlson_hx_diabetes_uncomp, operator: "==", value: 1}
    """
    raw = config.get("subgroups", config)
    if not isinstance(raw, Mapping):
        raise TypeError("Subgroups configuration must be a mapping.")

    definitions = []
    for name, rule in raw.items():
        if not isinstance(rule, Mapping):
            raise TypeError(f"Subgroup '{name}' must be a mapping.")

        conditions = rule.get("conditions", ())
        if conditions:
            if not isinstance(conditions, (list, tuple)) or not conditions:
                raise ValueError(f"Subgroup '{name}' conditions must be a non-empty list.")
            for condition in conditions:
                if not isinstance(condition, Mapping):
                    raise TypeError(f"Each condition in subgroup '{name}' must be a mapping.")
                missing = {"column", "operator", "value"} - set(condition)
                if missing:
                    raise ValueError(
                        f"Condition in subgroup '{name}' is missing: {sorted(missing)}"
                    )
            logic = str(rule.get("logic", "all")).lower()
            if logic not in {"all", "any"}:
                raise ValueError(f"Subgroup '{name}' logic must be 'all' or 'any'.")
            definitions.append(SubgroupDefinition(
                name=name,
                description=rule.get("description"),
                conditions=tuple(conditions),
                logic=logic,
            ))
            continue

        missing = {"column", "operator", "value"} - set(rule)
        if missing:
            raise ValueError(f"Subgroup '{name}' is missing: {sorted(missing)}")
        definitions.append(SubgroupDefinition(
            name=name,
            column=rule["column"],
            operator=rule["operator"],
            value=rule["value"],
            description=rule.get("description"),
        ))
    return definitions
