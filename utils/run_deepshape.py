import click
from selene_sdk.utils import load_path
from selene_sdk.utils import parse_configs_and_run

@click.command()
@click.argument('config_path')

def run_script(config_path):
    configs = load_path(config_path)
    parse_configs_and_run(configs)

if __name__ == '__main__':
    run_script()

# To run this script with the config_path set to /ABSOLUTE/PATH/config/train_deepshape.yml,
# you would execute the following command in your terminal:

# python run_deepshape.py /ABSOLUTE/PATH/config/train_deepshape.yml

