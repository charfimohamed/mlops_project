MLOPS project description - Binary classification of cats and dogs
==============================
Nojus Mickus s174447.
Mohamed Charfi s221586.
Ahmed Aziz Ben Haj Hmida s221551.
### Overall goal of the project

The goal of the project is to use convolutional neural network for binary classification of cats and dogs.


### What framework are you going to use (PyTorch Image Models, Transformer, Pytorch-Geometrics)

Since we are working with cat and dog images we will be using [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models) as our framework.


### How to you intend to include the framework into your project

We are going to use one of the pre-trained model from PyTorch Image Models.


### What data are you going to run on (initially, may change)

Initially, we plan to use cats and dogs [dataset](https://www.kaggle.com/datasets/alifrahman/dataset-for-wbc-classification) found on Kaggle. It contains 1000 examples each for training and 401 each for validation.


### What deep learning models do you expect to use

We are planning to use [Inception ResNet v2](https://arxiv.org/pdf/1602.07261.pdf) which builds on the Inception architecture but replaces the filter concatenation stage with residual connections. We will use the [implemented pre-trained model](https://huggingface.co/docs/timm/models/inception-resnet-v2) from PyTorch Image Models.

Get started
------------
1. Set up [Kaggle API](https://adityashrm21.github.io/Setting-Up-Kaggle/) on your local machine.
2. Clone the repo:
```
git clone https://github.com/charfimohamed/mlops_project
cd mlops_project
```
3. Create a new `conda` environment:
```
conda create -n mlops_project python=3.9
conda activate mlops_project
```
4. Install the dependencies
```
pip install -r requirements.txt
```
5. 
```
make data
```

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
