import re
from pathlib import Path
import pandas as pd


def get_target_domain(block: str) -> str:
    """Helper to determine domain from a YAML block. Defaults to 'problems'."""
    # Matches domain: 'vitals', domains: ['vitals'], etc.
    if re.search(r"domain[s]?:\s*.*?vitals", block, re.IGNORECASE):
        return "vitals"
    return "problems"


def generate_icare_table(yaml_path: Path, csv_paths: dict, phenotype_name: str) -> str:
    """Extracts codes from a phenotype definition and returns a Markdown table with debug output."""
    print(f"\n[DEBUG] Processing phenotype: '{phenotype_name}'")

    if not yaml_path.exists():
        print(f"[ERROR] YAML file not found at: {yaml_path.absolute()}")
        return ""

    text = yaml_path.read_text(encoding='utf-8')

    # 1. Match phenotype block
    p_pattern = rf"(?:^|\n){phenotype_name}:[\s\S]*?(?:\n[a-z_]+:|\Z)"
    p_match = re.search(p_pattern, text)
    if not p_match:
        print(f"[WARNING] Could not find phenotype block for '{phenotype_name}' in YAML.")
        return ""

    block = p_match.group(0)

    # 2. Extract codes list
    codes_match = re.search(r"\bcode[s]?:\s*\[(.*?)\]", block, re.DOTALL)
    if not codes_match:
        print(f"[WARNING] Could not find 'codes: [...]' inside block for '{phenotype_name}'.")
        return ""

    codes_str = codes_match.group(1)
    codes = re.findall(r"['\"]([^'\"]+)['\"]", codes_str)
    print(f"[DEBUG] Extracted codes for {phenotype_name}: {codes}")

    if not codes:
        print(f"[WARNING] Extracted code list was empty.")
        return ""

    # Determine Domain & Select CSV
    domain = get_target_domain(block)
    target_csv_path = csv_paths.get(domain, csv_paths['problems'])

    if not target_csv_path.exists():
        print(f"[ERROR] CSV file not found at: {target_csv_path.absolute()}")
        return ""

    # 3. Load CSV and match codes
    try:
        df = pd.read_csv(target_csv_path)
        df['code'] = df['code'].astype(str)
    except Exception as e:
        print(f"[ERROR] Failed to read {target_csv_path}: {e}")
        return ""

    matched = df[df['code'].isin(codes)].drop_duplicates(subset=['code'])
    print(f"[DEBUG] Found {len(matched)} matching rows in {target_csv_path.name} out of {len(codes)} codes searched.")

    if matched.empty:
        print(f"[NOTICE] Codes {codes} did not match any rows in {target_csv_path.name}.")
        return f"*No matching entries found in `{target_csv_path.name}` for codes: {', '.join(codes)}*"

    # 4. Build Markdown table
    md = [
        "| Code | Name | Unit | Type | Total Occurrences |",
        "| :--- | :--- | :--- | :--- | ---:|"
    ]
    for _, row in matched.iterrows():
        # Using .get() for safety in case columns slightly differ between tables
        name = row.get('name', 'N/A')
        unit = row.get('unit', 'none')
        ctype = row.get('type', 'N/A')
        count = row.get('total_occurrences', 0)
        md.append(f"| `{row['code']}` | {name} | {unit} | {ctype} | {count:,} |")

    print(f"[SUCCESS] Table generated successfully for '{phenotype_name}'.")
    return "\n".join(md)


def process_markdown_files():
    yaml_path = Path("/app/src/icare_risk/config/icare/phenotypes.yaml")
    docs_dir = Path("/app/docs")

    csv_paths = {
        "problems": Path("/app/data/lookups/standard/icareP.csv"),
        "vitals": Path("/app/data/lookups/standard/icareV.csv")
    }

    if not docs_dir.exists():
        print(f"[ERROR] 'docs/' directory not found at {docs_dir.absolute()}")
        return

    pattern = r"<!--\s*icare_table:\s*([a-zA-Z0-9_]+)\s*-->"
    files_updated = 0

    for md_file in docs_dir.glob("**/*.md"):
        content = md_file.read_text(encoding='utf-8')

        def replace_tag(match):
            phenotype_name = match.group(1)
            table_md = generate_icare_table(yaml_path, csv_paths, phenotype_name)
            if table_md:
                return f"\n\n{table_md}\n\n"
            else:
                return f"\n\n*Could not generate table for `{phenotype_name}`.*\n\n"

        new_content = re.sub(pattern, replace_tag, content)
        if new_content != content:
            md_file.write_text(new_content, encoding='utf-8')
            print(f"[INFO] Updated markdown file: {md_file}")
            files_updated += 1

    print(f"\n[DONE] Processing complete. Total markdown files updated: {files_updated}")


def generate_snippets():
    REPO_ROOT = Path(__file__).resolve().parents[3]

    yaml_path = REPO_ROOT / "src/icare_risk/config/icare/phenotypes.yaml"

    csv_paths = {
        "problems": REPO_ROOT / "data/lookups/standard/icareP.csv",
        "vitals": REPO_ROOT / "data/lookups/standard/icareV.csv"
    }

    snippets_dir = REPO_ROOT / "docs" / "_snippets"
    snippets_dir.mkdir(parents=True, exist_ok=True)

    if not yaml_path.exists():
        print("[ERROR] phenotypes.yaml not found in root.")
        return

    text = yaml_path.read_text(encoding='utf-8')

    # Load available dataframes into memory
    dfs = {}
    for domain, path in csv_paths.items():
        if path.exists():
            df = pd.read_csv(path)
            df['code'] = df['code'].astype(str)
            dfs[domain] = df
        else:
            print(f"[WARNING] CSV for '{domain}' not found at {path}")

    # Extract all top-level phenotype keys
    phenotype_names = re.findall(r"^([a-zA-Z0-9_]+):\s*(?:\n|\r\n)", text, re.MULTILINE)
    print(f"Found {len(phenotype_names)} phenotypes in YAML.")

    for phenotype_name in phenotype_names:
        p_pattern = rf"(?:^|\n){phenotype_name}:[\s\S]*?(?:\n[a-z_]+:|\Z)"
        p_match = re.search(p_pattern, text)
        if not p_match:
            continue

        block = p_match.group(0)

        # Identify domain to look up
        domain = get_target_domain(block)

        codes_match = re.search(r"\bcode[s]?:\s*\[(.*?)\]", block, re.DOTALL)
        if not codes_match:
            continue

        codes = re.findall(r"['\"]([^'\"]+)['\"]", codes_match.group(1))
        if not codes:
            continue

        # Fetch the correct DataFrame
        df = dfs.get(domain)

        # Build Markdown table snippet
        md = [
            f"| Code | Name | Unit | Type | Total Occurrences |",
            f"| :--- | :--- | :--- | :--- | ---:|"
        ]

        if df is None:
            md.append(f"*Data source for `{domain}` is missing. Could not search codes: {', '.join(codes)}*")
        else:
            matched = df[df['code'].isin(codes)].drop_duplicates(subset=['code'])
            if matched.empty:
                md.append(f"*No matching entries found in `{domain}` for codes: {', '.join(codes)}*")
            else:
                for _, row in matched.iterrows():
                    name = row.get('name', 'N/A')
                    unit = row.get('unit', 'none')
                    ctype = row.get('type', 'N/A')
                    count = row.get('total_occurrences', 0)
                    md.append(f"| `{row['code']}` | {name} | {unit} | {ctype} | {count:,} |")

        snippet_file = snippets_dir / f"{phenotype_name}.md"
        snippet_file.write_text("\n".join(md), encoding='utf-8')

    print(f"Successfully generated all snippets in `{snippets_dir}/` folder!")


if __name__ == "__main__":
    # process_markdown_files()
    generate_snippets()