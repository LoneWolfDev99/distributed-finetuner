import logging
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

from e2enetworks.cloud import tir

from helpers import download_dataset, download_folder_from_repo


if __name__ == "__main__":
    tir.init()

    if os.environ['dataset_type'] == "eos-bucket":
        dataset_id = int(os.environ['dataset_id'])
        dataset_path = os.environ['dataset_path']
        dataset_path = download_dataset(dataset_id, dataset_path)

    source_model_repo_id = int(os.environ['source_model_repo_id'])
    source_model_path = os.environ['source_model_path']
    if source_model_repo_id:
        download_folder_from_repo(source_model_repo_id, source_model_path)
        download_folder_from_repo(source_model_repo_id, 'base_model/')
