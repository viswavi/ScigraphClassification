# ScigraphClassification

Data was taken from this repo https://github.com/kimiyoung/planetoid

To train and evaluate our proposed models, simply run:
```
python package/main.py --dataset-name {cora/pubmed/citeseer} --dataset-style ind --model {t-crf/mlp/g-mlp/t-crf}
```

To train and evaluate the external baselines we use (SGC and Planetoid), see their associated submodules in this repository.
