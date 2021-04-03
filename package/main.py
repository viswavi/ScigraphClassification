import click
import os

from datasets.dataset import Dataset


@click.command()
@click.option('--data-directory', default=f'{os.getcwd()}/data', required=False)
@click.option('--dataset-name', type=click.Choice(['citeseer', 'pubmed', 'cora'], case_sensitive=False), required=True)
@click.option('--dataset-style', type=click.Choice(['ind', 'trans'], case_sensitive=False), required=True)
def main(data_directory, dataset_name, dataset_style):
    d = Dataset(data_directory, dataset_name, dataset_style)


if __name__ == "__main__":
    main()
