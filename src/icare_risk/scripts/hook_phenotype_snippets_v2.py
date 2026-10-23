"""
Phenotype Snippet Generator
===========================
Extracts clinical codes from phenotype YAML configurations, queries configured lookup databases,
and renders Markdown snippet tables.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 1. Explicit Domain-to-Lookup Mapping
# ----------------------------------------------------------------------

DOMAIN_LOOKUP_MAP: dict[str, list[str]] = {
    "problems": ["problems", "icd10", "res195"],  # Query icarePRO, ICD10, and res195 for problems
    "diagnoses": ["diagnoses", "icd10", "res195"],
    "vitals": ["vitals"],                         # Query icareVIT
    "pathology": ["pathology"],                   # Query icarePAT
    "prescribing": ["prescribing"],               # Query icarePHA
}


# ----------------------------------------------------------------------
# Configuration & Data Models
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class LookupSourceSpec:
    """Configuration specification for a lookup database CSV source."""
    name: str
    csv_path: Path
    code_col: str = "code"
    name_col: str = "name"
    yaml_keys: tuple[str, ...] = field(default_factory=tuple)
    extra_cols: tuple[str, ...] = ("unit", "type", "total_occurrences")
    allow_prefix_match: bool = False  # Toggle prefix search (e.g. True for icd10 / res195)


@dataclass(frozen=True)
class ExtractedCode:
    """Represents a code extracted from a phenotype definition."""
    code: str
    source_tag: str
    domain: str | None = None


@dataclass(frozen=True)
class CodeMatchResult:
    """Standardized result for a matched database record."""
    source: str
    code: str
    name: str
    extra_fields: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Code Extractor Engine
# ----------------------------------------------------------------------

class CodeExtractor:
    """Extracts codes and maps them explicitly to lookup sources via DOMAIN_LOOKUP_MAP."""

    def __init__(
        self,
        sources: Sequence[LookupSourceSpec],
        domain_map: dict[str, list[str]] = DOMAIN_LOOKUP_MAP,
    ):
        self.domain_map = domain_map
        self.key_to_source: dict[str, str] = {}
        for spec in sources:
            for key in spec.yaml_keys:
                self.key_to_source[key.lower()] = spec.name

    def extract(self, phenotype_def: dict[str, Any]) -> list[ExtractedCode]:
        extracted: set[ExtractedCode] = set()

        top_domains = phenotype_def.get("domains") or []
        default_domain = top_domains[0] if top_domains else "problems"
        kwargs = phenotype_def.get("kwargs", {})

        def resolve_sources(key_or_domain: str | None) -> list[str]:
            if not key_or_domain:
                key_or_domain = default_domain

            normalized = str(key_or_domain).lower()

            if normalized in self.key_to_source:
                return [self.key_to_source[normalized]]

            return self.domain_map.get(normalized, [normalized])

        def add_codes(key_or_domain: str | None, domain_hint: str | None, code_list: Any):
            sources = resolve_sources(key_or_domain)
            target_domain = domain_hint or default_domain

            if isinstance(code_list, (str, int)):
                codes = [str(code_list)]
            elif isinstance(code_list, list):
                codes = [str(c) for c in code_list if c is not None]
            else:
                return

            for c in codes:
                for src in sources:
                    extracted.add(ExtractedCode(code=c, source_tag=src, domain=target_domain))

        # Composite rules
        if "components" in kwargs and isinstance(kwargs["components"], list):
            for comp in kwargs["components"]:
                c_domain = comp.get("domain") or (comp.get("target_domains", [None])[0])
                if "codes" in comp:
                    add_codes(c_domain, c_domain, comp["codes"])

        # Multimodal rules
        for rule_key in ("code_rules", "expression_rules", "keyword_rules", "rules"):
            if rule_key in kwargs and isinstance(kwargs[rule_key], list):
                for rule in kwargs[rule_key]:
                    r_domain = rule.get("domain") or (rule.get("target_domains", [None])[0])
                    if "codes" in rule:
                        add_codes(r_domain, r_domain, rule["codes"])

        # Direct kwargs
        for key, val in kwargs.items():
            if key == "codes" or key.endswith("_codes"):
                key_prefix = key.removesuffix("_codes") if key.endswith("_codes") else key
                domain_ctx = kwargs.get("domain", default_domain) if key == "codes" else key_prefix
                add_codes(domain_ctx, default_domain, val)

        return sorted(list(extracted), key=lambda x: (x.source_tag, x.code))


# ----------------------------------------------------------------------
# Lookup Repository
# ----------------------------------------------------------------------

class LookupRepository:
    """Queries configured CSV database sources with optional dot-removal and prefix matching."""

    def __init__(self, sources: Sequence[LookupSourceSpec]):
        self.sources: dict[str, LookupSourceSpec] = {s.name: s for s in sources}
        self.dfs: dict[str, pd.DataFrame] = {}
        self._load_sources()

    def _load_sources(self):
        for name, spec in self.sources.items():
            if not spec.csv_path.exists():
                logger.warning(f"CSV path for source '{name}' not found: {spec.csv_path}")
                continue

            try:
                df = pd.read_csv(spec.csv_path)
                if spec.code_col not in df.columns:
                    logger.error(f"Column '{spec.code_col}' not found in {spec.csv_path}")
                    continue

                df[spec.code_col] = df[spec.code_col].astype(str).str.strip()

                # Precompute normalized code (_clean_code) without dots for prefix/exact comparisons
                df["_clean_code"] = (
                    df[spec.code_col]
                    .astype(str)
                    .str.replace(".", "", regex=False)
                    .str.strip()
                    .str.upper()
                )
                self.dfs[name] = df
            except Exception as e:
                logger.error(f"Failed to read CSV for '{name}' at {spec.csv_path}: {e}")

    @staticmethod
    def _deduplicate_matches(
            matched_rows: pd.DataFrame,
            spec: LookupSourceSpec,
    ) -> pd.DataFrame:
        """
        Keep one row per normalized code.

        When duplicate codes exist, prefer the first row with a
        non-empty name/description. Otherwise keep the first row.
        """

        rows = []

        for _, group in matched_rows.groupby(
                "_clean_code",
                sort=False,
                dropna=False,
        ):
            valid_names = group[
                group[spec.name_col].notna()
                & group[spec.name_col].astype(str).str.strip().ne("")
                & group[spec.name_col].astype(str).str.lower().ne("nan")
                ]

            if not valid_names.empty:
                rows.append(valid_names.iloc[0])
            else:
                rows.append(group.iloc[0])

        return pd.DataFrame(rows)

    def query(
            self,
            extracted_codes: Sequence[ExtractedCode]
    ) -> list[CodeMatchResult]:
        results: list[CodeMatchResult] = []

        for ec in extracted_codes:
            df = self.dfs.get(ec.source_tag)
            spec = self.sources.get(ec.source_tag)

            matched = False
            if df is not None and not df.empty and spec is not None:
                # Clean input target code (e.g. M05.0 or M05 -> M05)
                target_code = ec.code.replace(".", "").strip().upper()

                # Filter using startswith or exact equality depending on spec.allow_prefix_match
                if spec.allow_prefix_match:
                    matched_rows = df[df["_clean_code"].str.startswith(target_code)]
                else:
                    matched_rows = df[df["_clean_code"] == target_code]

                if not matched_rows.empty:
                    # Keep only one row per code.
                    matched_rows = self._deduplicate_matches(
                        matched_rows,
                        spec,
                    )
                    # Append all expanded matches into the final result table
                    for _, row in matched_rows.iterrows():
                        code_val = str(row.get(spec.code_col, ec.code))
                        name_val = str(row.get(spec.name_col, "N/A"))
                        extra_fields = {col: row[col] for col in spec.extra_cols if col in row}

                        results.append(
                            CodeMatchResult(
                                source=ec.source_tag,
                                code=code_val,
                                name=name_val,
                                extra_fields=extra_fields,
                            )
                        )
                    matched = True

            if not matched:
                results.append(
                    CodeMatchResult(
                        source=ec.source_tag,
                        code=ec.code,
                        name="*[No match found in lookup database]*",
                        extra_fields={},
                    )
                )

        return results


# ----------------------------------------------------------------------
# Table Renderer
# ----------------------------------------------------------------------

class TableRenderer:
    """Renders results into standard Markdown tables."""

    @staticmethod
    def render(
        matches: Sequence[CodeMatchResult],
        extracted_codes: Sequence[ExtractedCode],
        split_by_source: bool = False,
        show_occurrences: bool = True,
        sort_results: bool = True,
    ) -> str:
        if not matches and not extracted_codes:
            return "*No codes configured for this phenotype.*"

        row_list = list(matches)
        if sort_results:
            row_list.sort(key=lambda m: (m.source, m.code))

        has_unit = any("unit" in m.extra_fields for m in row_list)
        has_type = any("type" in m.extra_fields for m in row_list)
        has_count = show_occurrences and any("total_occurrences" in m.extra_fields for m in row_list)

        headers = ["#", "Source", "Code", "Name"]
        alignments = [":---", ":---", ":---", ":---"]

        if has_unit:
            headers.append("Unit")
            alignments.append(":---")
        if has_type:
            headers.append("Type")
            alignments.append(":---")
        if has_count:
            headers.append("Total Occurrences")
            alignments.append("---:")

        header_line = "| " + " | ".join(headers) + " |"
        align_line = "| " + " | ".join(alignments) + " |"

        md_lines = [header_line, align_line]

        for idx, match in enumerate(row_list, start=1):
            row_cells = [str(idx), f"`{match.source}`", f"`{match.code}`", match.name]
            if has_unit:
                row_cells.append(str(match.extra_fields.get("unit", "N/A")))
            if has_type:
                row_cells.append(str(match.extra_fields.get("type", "N/A")))
            if has_count:
                count_val = match.extra_fields.get("total_occurrences", "N/A")
                formatted_count = f"{count_val:,}" if isinstance(count_val, (int, float)) else str(count_val)
                row_cells.append(formatted_count)

            md_lines.append("| " + " | ".join(row_cells) + " |")

        table_md = "\n".join(md_lines)
        wrapper_class = "phenotype-table-split" if split_by_source else "phenotype-table"
        return f'\n\n<div class="{wrapper_class}">\n\n{table_md}\n\n</div>\n\n'


# ----------------------------------------------------------------------
# Pipeline Orchestrator
# ----------------------------------------------------------------------

class PhenotypeSnippetPipeline:
    def __init__(self, yaml_path: Path,
                       snippets_dir: Path,
                       sources: Sequence[LookupSourceSpec]):
        self.yaml_path = yaml_path
        self.snippets_dir = snippets_dir
        self.sources = sources
        self.extractor = CodeExtractor(sources, domain_map=DOMAIN_LOOKUP_MAP)
        self.repository = LookupRepository(sources)

    def run(self, split_by_source: bool = False, show_total_occurrences: bool = False):
        if not self.yaml_path.exists():
            logger.error(f"phenotypes.yaml not found at {self.yaml_path}")
            return

        self.snippets_dir.mkdir(parents=True, exist_ok=True)

        with open(self.yaml_path, "r", encoding="utf-8") as f:
            try:
                phenotypes = yaml.safe_load(f)
            except yaml.YAMLError as e:
                logger.error(f"Failed to parse YAML file: {e}")
                return

        generated_count = 0

        for name, config in phenotypes.items():
            if not isinstance(config, dict):
                continue

            extracted_codes = self.extractor.extract(config)
            if not extracted_codes:
                continue

            matches = self.repository.query(extracted_codes)

            table_md = TableRenderer.render(
                matches=matches,
                extracted_codes=extracted_codes,
                split_by_source=split_by_source,  # Set to True for separate tables per source
                show_occurrences=show_total_occurrences,
                sort_results=True,
            )

            snippet_file = self.snippets_dir / f"{name}.md"
            snippet_file.write_text(table_md, encoding="utf-8")
            generated_count += 1

        logger.info(f"Successfully generated {generated_count} snippet files in `{self.snippets_dir}`!")


def main():
    REPO_ROOT = Path(__file__).resolve().parents[3]

    yaml_path = REPO_ROOT / "src/icare_risk/config/icare/phenotypes.yaml"
    snippets_dir = REPO_ROOT / "docs" / "_snippets"
    data_dir = REPO_ROOT / "data/lookups/standard"

    lookup_sources = [
        LookupSourceSpec(name="problems", csv_path=data_dir / "icarePRO.csv",
                         code_col="code", name_col="name",
                         yaml_keys=("problems", "codes")),
        LookupSourceSpec(name="vitals", csv_path=data_dir / "icareVIT.csv",
                         code_col="code", name_col="name",
                         yaml_keys=("vitals",)),
        LookupSourceSpec(name="pathology", csv_path=data_dir / "icarePAT.csv",
                         code_col="code", name_col="name",
                         yaml_keys=("pathology",)),
        LookupSourceSpec(name="prescribing", csv_path=data_dir / "icarePHA.csv",
                         code_col="code", name_col="name",
                         yaml_keys=("prescribing",)),
        LookupSourceSpec(name="res195", csv_path=data_dir / "res195.csv",
                         code_col="code", name_col="description",
                         yaml_keys=("res195_codes", "res195"),
                         allow_prefix_match=True),
        LookupSourceSpec(name="icd10", csv_path=data_dir / "ICD10.csv",
                         code_col="Code", name_col="Description",
                         yaml_keys=("icd10_codes", "icd10"),
                         allow_prefix_match=True),
        LookupSourceSpec(name="diagICD10", csv_path=data_dir / "icareDIA.csv",
                         code_col="diagnosis_code_icd", name_col="diagnosis_desc_icd",
                         yaml_keys=("diagnosis", "codes"),
                         allow_prefix_match=True),
        LookupSourceSpec(name="diagSNOMED", csv_path=data_dir / "icareDIA.csv",
                         code_col="diagnosis_code_snomed", name_col="diagnosis_desc_snomed",
                         yaml_keys=("diagnosis", "codes")),
    ]

    pipeline = PhenotypeSnippetPipeline(
        yaml_path=yaml_path,
        snippets_dir=snippets_dir,
        sources=lookup_sources
    )

    pipeline.run(
        split_by_source=False,
        show_total_occurrences=False,
    )


if __name__ == "__main__":
    main()