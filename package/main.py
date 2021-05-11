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
@click.option('--seed', type=click.INT, required=False)
@click.option('--skip-parameter-search', is_flag=True)
@click.option('--ensemble', is_flag=True)
@click.option('--validate', is_flag=True)
def main(data_directory, dataset_name, dataset_style, model, seed, skip_parameter_search, ensemble, validate):
    if seed is None:
        seed = DEFAULT_SEED
    train_dataset = Dataset(data_directory, dataset_name, dataset_style, train=True)
    test_dataset = Dataset(data_directory, dataset_name, dataset_style, train=False)
    if model == 'brf':
        run_brf_experiments(train_dataset, test_dataset, validate)
    if model == 'mlp':
        run_mlp_experiments(train_dataset, test_dataset, validate)
    if model == 'g-mlp':
        run_graph_mlp_experiments(train_dataset, test_dataset, validate)
    if model == 't-crf':
        run_tree_crf_experiments(train_dataset, test_dataset, ensemble, skip_parameter_search=skip_parameter_search, seed=seed)


if __name__ == "__main__":
    main()
