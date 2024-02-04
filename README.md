project description - Binary classification of cats and dogs
==============================
### Overall goal of the project

The goal of the project is to use convolutional neural network for binary classification of cats and dogs 
The main focus of the project is to use the most famous machine learning tools (FastApi, training the model on Google cloud, github Actions etc..)

### Dataset:

we use cats and dogs [dataset](https://www.kaggle.com/datasets/alifrahman/dataset-for-wbc-classification) found on Kaggle. It contains 1000 examples each for training and 401 each for validation.


### used model: 

We use  ResNet-50 (https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html). We will use the pretrained model from torchvision.

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
5. download the data
```
make data  
```
or
```
dvc pull
```
6. train the model : navigate to src/models and run: 
```
python train_model.py
```
or (if you want to run the wandb sweep and profiling): navigate to src/models and run:
```
python optimize_train_model.py
```
7. test the model : navigate to src/models and run: 
```
python predict_model.py
```
or (if you want to test the model with quantization): navigate to src/models and run:
```
python predict_model_quantization.py
```
8. run the unit tests with coverage:
```
coverage run -m pytest tests/
```
and 
```
coverage report
```
9. create the API to deploy the model locally: navigate to app/ and run:
```
uvicorn --reload --port 8000 main:app
```
10. to test the model that we trained (97% test accuracy): 
(only works if our Cloud Run in GCP is still running)
```
curl -X POST -H "Content-Type: multipart/form-data" -F "data=@imagepath/image.png" https://inference-for-gcp-sjsexi6d7a-ew.a.run.app/cv_model/
```
where imagepath/image.png is the path to the image you want to classify.
you can also type in your favourite browser:  https://inference-for-gcp-sjsexi6d7a-ew.a.run.app/docs, click on try it out, upload your image and execute. you'll see the classification in the response


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
