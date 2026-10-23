import yaml
from typing import Any, Dict, Optional

def load_phenotypes_config(yaml_path: str, prefix: Optional[str] = None) -> Dict[str, Any]:
  """Loads a YAML configuration file and optionally filters keys by a prefix.

  Parameters
  ----------
  yaml_path : str
      Path to the YAML configuration file.
  prefix : str, optional
      Prefix string to filter configuration keys, by default None.

  Returns
  -------
  dict
      The filtered (or full) configuration dictionary.
  """
  with open(yaml_path, "r") as f:
    config = yaml.safe_load(f) or {}

  if prefix:
    return {k: v for k, v in config.items() if k.startswith(prefix)}
  return config