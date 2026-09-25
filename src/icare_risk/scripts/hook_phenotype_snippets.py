from pathlib import Path
from typing import Any
import pandas as pd
import yaml


class CodeExtractor:
    """Extracts (domain, code) pairs from parsed phenotype YAML configurations based on function strategies."""

    @staticmethod
    def extract(phenotype_def: dict[str, Any]) -> dict[str, list[str]]:
        """
        Parses a phenotype dictionary and returns a mapping of {domain: [codes]}.
        """
        domain_codes: dict[str, set[str]] = {}

        top_domains = phenotype_def.get("domains") or []
        default_domain = top_domains[0] if top_domains else "problems"

        kwargs = phenotype_def.get("kwargs", {})

        def add_code(domain: str | None, code_list: Any):
            target_domain = domain or default_domain
            if target_domain not in domain_codes:
                domain_codes[target_domain] = set()

            if isinstance(code_list, (str, int)):
                code_list = [code_list]
            elif not isinstance(code_list, list):
                return

            for c in code_list:
                domain_codes[target_domain].add(str(c))

        # Strategy 1: Composite rules with nested components (e.g., derive_composite_rules)
        if "components" in kwargs and isinstance(kwargs["components"], list):
            for comp in kwargs["components"]:
                c_domain = comp.get("domain") or (
                    comp.get("target_domains", [None])[0]
                )
                if "codes" in comp:
                    add_code(c_domain, comp["codes"])

        # Strategy 2: Multimodal rules (e.g., code_rules, expression_rules)
        for rule_key in ("code_rules", "expression_rules", "keyword_rules"):
            if rule_key in kwargs and isinstance(kwargs[rule_key], list):
                for rule in kwargs[rule_key]:
                    r_domain = rule.get("domain") or (
                        rule.get("target_domains", [None])[0]
                    )
                    if "codes" in rule:
                        add_code(r_domain, rule["codes"])

        # Strategy 3: Standard single or multi-code kwargs (e.g., codes, wbc_codes, rr_codes)
        for key, val in kwargs.items():
            if key == "codes" or key.endswith("_codes"):
                # Infer domain from top-level if not overridden
                add_code(default_domain, val)

        return {domain: sorted(list(codes)) for domain, codes in domain_codes.items()}


class LookupDatabase:
    """Loads domain CSVs into memory and queries code matches."""

    def __init__(self, csv_paths: dict[str, Path]):
        self.dfs: dict[str, pd.DataFrame] = {}
        self._load_csvs(csv_paths)

    def _load_csvs(self, csv_paths: dict[str, Path]):
        for domain, path in csv_paths.items():
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    df["code"] = df["code"].astype(str)
                    self.dfs[domain] = df
                except Exception as e:
                    print(f"[ERROR] Failed to read CSV for domain '{domain}' at {path}: {e}")
            else:
                print(f"[WARNING] CSV path for domain '{domain}' not found: {path}")

    def query(self, domain: str, codes: list[str]) -> pd.DataFrame:
        df = self.dfs.get(domain)
        if df is None or df.empty:
            return pd.DataFrame()

        matched = df[df["code"].isin(codes)].drop_duplicates(subset=["code"])
        return matched


class TableRenderer:
    """Renders dataframes into styled Markdown tables."""

    @staticmethod
    def render(
        matched_by_domain: dict[str, pd.DataFrame],
        requested_codes: dict[str, list[str]],
        show_total_occurrences: bool = True,
        sort_codes: bool = True,
    ) -> str:
        all_rows = []
        is_multi_domain = len(requested_codes) > 1

        for domain, codes in requested_codes.items():
            matched_df = matched_by_domain.get(domain)

            if matched_df is None or matched_df.empty:
                continue

            for _, row in matched_df.iterrows():
                row_data = {
                    "domain": domain,
                    "code": row.get("code", "N/A"),
                    "name": row.get("name", "N/A"),
                    "unit": row.get("unit", "none"),
                    "type": row.get("type", "N/A"),
                }
                if show_total_occurrences:
                    row_data["count"] = row.get("total_occurrences", 0)

                all_rows.append(row_data)

        if not all_rows:
            all_missing = [f"`{c}` ({d})" for d, cs in requested_codes.items() for c in cs]
            return f"*No matching database entries found for codes: {', '.join(all_missing)}*"

        # Sort rows by domain and code if enabled
        if sort_codes:
            all_rows.sort(key=lambda r: (r["domain"], r["code"]))

        # Build dynamic headers and column alignments
        headers = []
        align = []

        if is_multi_domain:
            headers.append("Domain")
            align.append("| :---")

        headers.extend(["Code", "Name", "Unit", "Type"])
        align.extend(["| :---", "| :---", "| :---", "| :---"])

        if show_total_occurrences:
            headers.append("Total Occurrences")
            align.append("| ---:|")

        md = ["| " + " | ".join(headers) + " |", " ".join(align)]

        for r in all_rows:
            row_cells = []
            if is_multi_domain:
                row_cells.append(f"`{r['domain']}`")

            row_cells.extend([f"`{r['code']}`", str(r["name"]), str(r["unit"]), str(r["type"])])

            if show_total_occurrences:
                count_formatted = f"{r['count']:,}" if isinstance(r['count'], (int, float)) else r['count']
                row_cells.append(str(count_formatted))

            md.append("| " + " | ".join(row_cells) + " |")

        return "\n".join(md)


def generate_snippets(show_total_occurrences: bool = True, sort_codes: bool = True):
    REPO_ROOT = Path(__file__).resolve().parents[3]

    yaml_path = REPO_ROOT / "src/icare_risk/config/icare/phenotypes.yaml"
    snippets_dir = REPO_ROOT / "docs" / "_snippets"
    snippets_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = {
        "problems": REPO_ROOT / "data/lookups/standard/icarePRO.csv",
        "vitals": REPO_ROOT / "data/lookups/standard/icareVIT.csv",
        "pathology": REPO_ROOT / "data/lookups/standard/icarePAT.csv",
        "prescribing": REPO_ROOT / "data/lookups/standard/icarePHA.csv",
        #"microbiology": REPO_ROOT / "data/lookups/standard/icareMIC.csv",
    }

    if not yaml_path.exists():
        print(f"[ERROR] phenotypes.yaml not found at {yaml_path}")
        return

    # Parse entire YAML file into structured Python dictionary
    with open(yaml_path, "r", encoding="utf-8") as f:
        try:
            phenotypes = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"[ERROR] Failed to parse YAML file: {e}")
            return

    db = LookupDatabase(csv_paths)
    generated_count = 0

    for name, config in phenotypes.items():
        if not isinstance(config, dict):
            continue

        # Extract domain -> codes mapping
        domain_codes = CodeExtractor.extract(config)
        if not domain_codes:
            continue

        # Query database for each domain
        matched_by_domain = {}
        for domain, codes in domain_codes.items():
            matched_by_domain[domain] = db.query(domain, codes)

        # Render snippet table
        table_md = TableRenderer.render(
            matched_by_domain,
            domain_codes,
            show_total_occurrences=show_total_occurrences,
            sort_codes=sort_codes,
        )

        snippet_file = snippets_dir / f"{name}.md"
        snippet_file.write_text(table_md, encoding="utf-8")
        generated_count += 1

    print(f"[SUCCESS] Successfully generated {generated_count} snippet files in `{snippets_dir}`!")


if __name__ == "__main__":
    generate_snippets(
        show_total_occurrences=False,
        sort_codes=True
    )