import click
import os

from datasets.dataset import Dataset
from experiments.run_brf_experiments import run_brf_experiments
from experiments.run_mlp_experiments import run_mlp_experiments
from experiments.run_graph_mlp_experiments import run_graph_mlp_experiments
from experiments.run_tree_crf_experiments import run_tree_crf_experiments


@click.command()
@click.option('--data-directory', default=f'{os.getcwd()}/data', required=False)
@click.option('--dataset-name', type=click.Choice(['citeseer', 'pubmed', 'cora'], case_sensitive=False), required=True)
@click.option('--dataset-style', type=click.Choice(['ind', 'trans'], case_sensitive=False), required=True)
@click.option('--model', type=click.Choice(['brf', 'mlp', 'g-mlp', 't-crf'], case_sensitive=False), required=True)
def main(data_directory, dataset_name, dataset_style, model):
    train_dataset = Dataset(data_directory, dataset_name, dataset_style, train=True)
    test_dataset = Dataset(data_directory, dataset_name, dataset_style, train=False)
    if model == 'brf':
        run_brf_experiments(train_dataset, test_dataset)
    if model == 'mlp':
        run_mlp_experiments(train_dataset, test_dataset)
    if model == 'g-mlp':
        run_graph_mlp_experiments(train_dataset, test_dataset)
    if model == 't-crf':
        run_tree_crf_experiments(train_dataset, test_dataset)


if __name__ == "__main__":
    main()
