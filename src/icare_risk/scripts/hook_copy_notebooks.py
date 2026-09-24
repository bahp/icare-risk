import json
import shutil
import subprocess
from pathlib import Path

import json
import shutil
import subprocess
from pathlib import Path


def sanitize_notebook(ipynb_path: Path) -> None:
    """Strips JetBrains IDE non-standard metadata (e.g. 'jetTransient') from cell outputs."""
    with open(ipynb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    modified = False
    for cell in nb_data.get("cells", []):
        for output in cell.get("outputs", []):
            if isinstance(output, dict) and "jetTransient" in output:
                del output["jetTransient"]
                modified = True

    if modified:
        with open(ipynb_path, "w", encoding="utf-8") as f:
            json.dump(nb_data, f, indent=1)


def main(repo_root: Path | None = None, convert_to_md: bool = False) -> None:
    """
    Syncs tutorial notebooks to docs/.

    :param repo_root: Path to repository root.
    :param convert_to_md: If True, converts .ipynb to .md via nbconvert.
                          If False, retains raw .ipynb files for downloads.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]

    notebooks_src = repo_root / "notebooks" / "tutorials"
    notebooks_dst = repo_root / "docs" / "notebooks" / "tutorials"

    if notebooks_src.exists():
        if notebooks_dst.exists():
            shutil.rmtree(notebooks_dst)
        notebooks_dst.mkdir(parents=True, exist_ok=True)

        for ipynb in notebooks_src.glob("*.ipynb"):
            target_ipynb = notebooks_dst / ipynb.name
            shutil.copy2(ipynb, target_ipynb)

            if convert_to_md:
                # Clean JetBrains metadata before nbconvert runs
                sanitize_notebook(target_ipynb)

                print(f"[ASSET BUILDER] Converting {ipynb.name} to Markdown...")
                subprocess.run([
                    "jupyter", "nbconvert",
                    "--to", "markdown",
                    "--TemplateExporter.exclude_input_prompt=False",
                    "--output-dir", str(notebooks_dst),
                    str(target_ipynb)
                ], check=True)

                # Remove raw .ipynb so only generated .md remains
                target_ipynb.unlink(missing_ok=True)
            else:
                print(f"[ASSET BUILDER] Copied raw notebook {ipynb.name}")

        status_msg = "converted to Markdown" if convert_to_md else "copied as raw .ipynb files"
        print(f"[ASSET BUILDER] Notebooks successfully {status_msg} in {notebooks_dst}")

def on_config(config, **kwargs):
    """MkDocs / Zensical lifecycle hook entry point."""
    config_file = getattr(config, "config_file_path", None) or (
        config.get("config_file_path") if isinstance(config, dict) else None
    )
    repo_root = Path(config_file).parent if config_file else Path(__file__).resolve().parents[3]
    main(repo_root)
    return config

if __name__ == "__main__":
    main(convert_to_md=False)