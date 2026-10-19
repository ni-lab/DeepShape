"""Resolve documented repository/data/output variables without cluster-specific paths."""
import os
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]

def expanded_config_text(path):
    path = Path(path).expanduser().resolve()
    values = dict(os.environ)
    values.setdefault('DEEPSHAPE_ROOT', str(ROOT))
    values.setdefault('DEEPSHAPE_DATA', str(ROOT/'data'))
    values.setdefault('DEEPSHAPE_OUTPUT', str(ROOT/'quickstart-output'))
    text = path.read_text()
    if '/ABSOLUTE/PATH' in text or '/path/to/' in text:
        raise ValueError('Replace placeholder paths in ' + str(path) + '; see the configuration examples in README.md.')
    try:
        return Template(text).substitute(values)
    except KeyError as error:
        raise ValueError('Set the configuration environment variable ' + str(error.args[0])) from error
