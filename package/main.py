import click
import os

from datasets.dataset import Dataset
from experiments.run_brf_experiments import run_brf_experiments


@click.command()
@click.option('--data-directory', default=f'{os.getcwd()}/data', required=False)
@click.option('--dataset-name', type=click.Choice(['citeseer', 'pubmed', 'cora'], case_sensitive=False), required=True)
@click.option('--dataset-style', type=click.Choice(['ind', 'trans'], case_sensitive=False), required=True)
@click.option('--model', type=click.Choice(['brf', 'mlp'], case_sensitive=False), required=True)
def main(data_directory, dataset_name, dataset_style, model):
    train_dataset = Dataset(data_directory, dataset_name, dataset_style, train=True)
    test_dataset = Dataset(data_directory, dataset_name, dataset_style, train=False)
    if model == 'brf':
        run_brf_experiments(train_dataset, test_dataset)


if __name__ == "__main__":
    main()
