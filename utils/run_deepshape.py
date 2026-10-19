"""Run a Selene training/evaluation configuration from any working directory."""
import argparse
import os
from pathlib import Path
import tempfile
from config_paths import expanded_config_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config_path', type=Path)
    args = parser.parse_args()
    text = expanded_config_text(args.config_path)
    from selene_sdk.utils import load_path, parse_configs_and_run
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as stream:
            stream.write(text)
            temporary = stream.name
        configs = load_path(temporary)
        parse_configs_and_run(configs)
    finally:
        if temporary:
            os.unlink(temporary)

if __name__ == '__main__':
    main()
