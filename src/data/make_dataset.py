# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
import PIL
from dotenv import find_dotenv, load_dotenv

load_dotenv()

import kaggle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

kaggle_username = os.environ.get('KAGGLE_USERNAME')
kaggle_key = os.environ.get('KAGGLE_KEY')


class CatDogDataset(Dataset):
    def __init__(self, split: str, in_folder: str, out_folder: str, transform=None):

        super().__init__()

        self.in_folder = in_folder
        self.out_folder = out_folder

        if not (in_folder / "data").exists():
            print("Downloading data from Kaggle")
            self.download_raw_data(in_folder)

        self.split = split
        self.df = self.prepare_dataframe(split)
        self.file_names = self.df["filename"].values
        self.labels = self.df["label"].values
        self.category = self.df["category"].values

        self.transform = transform

    def download_raw_data(self, download_path: Path):
        """
        Downloads raw data from Kaggle.
        Make sure to setup your access token using https://adityashrm21.github.io/Setting-Up-Kaggle/
        """
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files("alifrahman/dataset-for-wbc-classification", path=download_path, unzip=True)

    def prepare_dataframe(self, split: str):
        categories = []
        filenames = []
        SEED = 42
        le = preprocessing.LabelEncoder()

        if split == "train":
            for dogfile in os.listdir(self.in_folder / "data" / "train" / "dogs"):
                categories.append("dogs")
                filenames.append(dogfile)
            for catfile in os.listdir(self.in_folder / "data" / "train" / "cats"):
                categories.append("cats")
                filenames.append(catfile)

            df_train = pd.DataFrame({"filename": filenames, "category": categories})
            df_train["label"] = le.fit_transform(df_train["category"])
            df_train = df_train.sample(frac=1, random_state=SEED).reset_index(drop=True)
            return df_train

        else:
            for dogfile in os.listdir(self.in_folder / "data" / "validation" / "dogs"):
                categories.append("dogs")
                filenames.append(dogfile)
            for catfile in os.listdir(self.in_folder / "data" / "validation" / "cats"):
                categories.append("cats")
                filenames.append(catfile)

            df_val = pd.DataFrame({"filename": filenames, "category": categories})

            le = preprocessing.LabelEncoder()
            df_val["label"] = le.fit_transform(df_val["category"])
            df_val = df_val.sample(frac=1, random_state=SEED).reset_index(drop=True)

            df_val, df_test = train_test_split(df_val, test_size=0.5, random_state=SEED)
            df_val.reset_index(drop=True, inplace=True)
            df_test.reset_index(drop=True, inplace=True)

        if split == "validation":
            return df_val
        elif split == "test":
            return df_test
        else:
            print('split can be either "train", "validation" or "test"')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        folder = "train" if self.split == "train" else "validation"
        img = PIL.Image.open(self.in_folder / "data" / folder / self.category[idx] / self.file_names[idx])

        if self.transform:
            img = self.transform(img)

        img = np.asarray(img)
        return img, self.labels[idx]


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    input_filepath, output_filepath = Path(input_filepath), Path(output_filepath)
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    image_size = 128
    data_resize = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    train_dataset = CatDogDataset(
        split="train",
        in_folder=input_filepath,
        out_folder=output_filepath,
        transform=data_resize,
    )
    validation_dataset = CatDogDataset(
        split="validation",
        in_folder=input_filepath,
        out_folder=output_filepath,
        transform=data_resize,
    )
    test_dataset = CatDogDataset(
        split="test",
        in_folder=input_filepath,
        out_folder=output_filepath,
        transform=data_resize,
    )
    print(f"train dataset size is : {len(train_dataset)}")
    print(f"validation dataset size is : {len(validation_dataset)}")
    print(f"test dataset size is : {len(test_dataset)}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    (project_dir / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (project_dir / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (project_dir / "data" / "interim").mkdir(parents=True, exist_ok=True)
    (project_dir / "data" / "external").mkdir(parents=True, exist_ok=True)
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
