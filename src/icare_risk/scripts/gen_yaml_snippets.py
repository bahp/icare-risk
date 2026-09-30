"""
Standalone YAML Snippet Generator
=================================
Loads phenotypes.yaml (resolving all YAML anchors/aliases automatically via safe_load),
and writes out individual .yaml snippet files for each phenotype entry (including all
charlson_hx_* entries) into docs/_snippets/yaml/ so they can be included via --8<--.
"""

from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    REPO_ROOT = Path(__file__).resolve().parents[3]  # Adjust based on your script location relative to root
    yaml_path = REPO_ROOT / "src/icare_risk/config/icare/phenotypes.yaml"
    output_dir = REPO_ROOT / "docs" / "_snippets" / "yaml"

    if not yaml_path.exists():
        logger.error(f"Phenotypes YAML not found at {yaml_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(yaml_path, "r", encoding="utf-8") as f:
        try:
            # PyYAML's safe_load automatically expands all YAML anchors (&) and aliases (*)
            phenotypes = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML: {e}")
            return

    count = 0
    for name, config in phenotypes.items():
        if not isinstance(config, dict):
            continue

        snippet_data = {name: config}
        snippet_file = output_dir / f"{name}.yaml"

        with open(snippet_file, "w", encoding="utf-8") as out:
            yaml.dump(
                snippet_data,
                out,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True
            )

        count += 1

    logger.info(f"Successfully generated {count} expanded YAML snippet files in `{output_dir}`!")


if __name__ == "__main__":
    main()