from pathlib import Path
import importlib.resources
import shutil


def main():
    """Copies bundled pipeline notebooks from the installed icare-risk package to the current working directory."""
    target_path = Path.cwd()

    try:
        # Securely access the bundled notebooks directory inside the installed package
        notebooks_ref = importlib.resources.files("icare_risk").joinpath("notebooks")
    except (ModuleNotFoundError, FileNotFoundError) as e:
        print(f"Error: Could not locate bundled notebooks directory in the installed package: {e}")
        return

    if not notebooks_ref.is_dir():
        print("Error: Bundled notebooks path is not a directory.")
        return

    count = 0
    for item in notebooks_ref.iterdir():
        if item.is_file() and item.name.endswith(".ipynb"):
            dest_file = target_path / item.name
            if dest_file.exists():
                print(f"Skipping {item.name} (already exists in current directory)")
                continue

            dest_file.write_bytes(item.read_bytes())
            print(f"Copied: {item.name}")
            count += 1

    print(f"\nSuccessfully extracted {count} notebooks to {target_path.resolve()}")


if __name__ == "__main__":
    main()