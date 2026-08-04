# 📖 Guide: Running `icare_risk` Entrypoints in Azure ML

When running custom CLI entrypoints (like `icare-risk-generate`) inside Azure ML, Jupyter notebooks and the terminal often have conflicting environment paths. This guide provides solutions for resolving these conflicts depending on how you prefer to run the code.

---

## 1. Running purely in Python (Recommended)

Jupyter kernels automatically inject background connection parameters into `sys.argv`. This causes `argparse` to crash with a `SystemExit 2` error because it does not recognize the Jupyter background flags.

Choose one of these fixes to run the script cleanly inside a notebook cell.

### Option A: The Developer Standard (Requires editing `a_generate_data.py`)

Update your script's `main()` function to accept optional arguments. This prevents it from reading Jupyter's background data.

**In `a_generate_data.py`:**

```python
def main(args=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument(...)

    # Pass the args parameter here
    parsed_args = parser.parse_args(args)
```

**In your Jupyter Notebook:**

```python
from icare_risk.scripts.a_generate_data import main

# Run with empty arguments (bypasses Jupyter's sys.argv)
main([])

# Or pass CLI flags as a list
main(['--config', 'config.yaml', '--limit', '100'])
```

---

### Option B: The Quick Hack (No script edits needed)

Override `sys.argv` temporarily before calling the function.

**In your Jupyter Notebook:**

```python
import sys
from icare_risk.scripts.a_generate_data import main

# Reset sys.argv. Add CLI flags to this list if needed.
sys.argv = ['icare-risk-generate']

main()
```

---

## 2. Running via Notebook Subshell (`!`)

Notebook subshells (commands prefixed with `!`) do not automatically inherit the Jupyter kernel's Python path. This can result in `command not found` errors.

### Option A: Align the Path

Put this in your first notebook setup cell to map the subshell `PATH` to your active Jupyter kernel.

**In your Jupyter Notebook:**

```python
import os
import sys
from pathlib import Path

kernel_bin_dir = Path(sys.executable).parent
os.environ['PATH'] = f"{kernel_bin_dir}:{os.environ['PATH']}"
```

Once executed, you can run the command normally anywhere else in the notebook:

```bash
!icare-risk-generate
```

### Option B: Target the Executable Directly

Force the subshell to use the active kernel's Python engine without modifying the `PATH`.

**In your Jupyter Notebook:**

```python
import sys

!{sys.executable} -m icare_risk.scripts.a_generate_data
```

---

## 3. Running from the Azure ML Terminal

The Azure ML terminal defaults to the base Conda environment upon opening. It may miss local package installations or specific kernels.

If `icare-risk-generate` returns `command not found` in the terminal, use one of the following methods.

### Option A: Activate the Correct Conda Environment

Switch to the environment where your package is installed.

```bash
conda env list
conda activate <your-environment-name>  # e.g., azureml_py38

icare-risk-generate
```

### Option B: Add Your Local `bin` Directory to `PATH`

If you installed the package via `pip install --user` or `pip install -e .` without an active Conda environment, the executable may be placed in your local user directory.

```bash
export PATH="$HOME/.local/bin:$PATH"

icare-risk-generate
```

**Tip:** To make this permanent, append the following line to your `~/.bashrc` file:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

---

## Quick Reference

| Where you are running          | Recommended approach                   |
| ------------------------------ | -------------------------------------- |
| Jupyter Notebook — Python      | Import `main()` and call `main([])`    |
| Jupyter Notebook — `!` command | Use `!{sys.executable} -m ...`         |
| Azure ML Terminal              | Activate the correct Conda environment |
| User-level pip installation    | Add `$HOME/.local/bin` to `PATH`       |

---
