from pathlib import Path
import icare_risk
import shutil


def main():
  """Copies bundled pipeline notebooks from the installed icare-risk package to the current working directory."""
  target_path = Path.cwd()

  # Resolve path directly from the installed package root (__file__)
  package_root = Path(icare_risk.__file__).resolve().parent
  source_dir = package_root / "notebooks"

  if not source_dir.is_dir():
    print(
        f"Error: Could not locate bundled notebooks directory at:"
        f" {source_dir}"
    )
    return

  count = 0
  for item in source_dir.glob("*.ipynb"):
    dest_file = target_path / item.name
    if dest_file.exists():
      print(f"Skipping {item.name} (already exists in current directory)")
      continue

    shutil.copy2(item, dest_file)
    print(f"Copied: {item.name}")
    count += 1

  print(f"\nSuccessfully extracted {count} notebooks to {target_path.resolve()}")


if __name__ == "__main__":
  main()