import itertools
import os
import random
import re
import string
import datasets

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from text_to_num import text2num
import torch
from torch.utils.data import DataLoader

################# Reproducibility ######################à
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False


# Using a generator and the following function as `worker_init_fn` preserves reproducibility when using DataLoader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_reproducible_dataloader(*args, **kwargs):
    generator = torch.Generator()
    return DataLoader(*args,
                      **kwargs,
                      worker_init_fn=seed_worker,
                      generator=generator)
###############################################################


def create_dirs_for_file(file_path):
    dir = os.path.dirname(file_path)
    ensure_dir_exists(dir)


def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def is_number(string: str) -> bool:
    """
    Check whether a string is a number, both written or numeric.

    Args:
    - string (str): The string to be checked.

    Returns:
    - True if the string is a number, False otherwise.
    """
    if string.isdigit():
        return True
    try:
        # Try to convert written number to integer
        text2num(string, 'en', relaxed=True)
        return True
    except ValueError:
        return False

def batched_function(fn, scalar_output=True):

    def execute_on_batch(batch):
        examples = [
            fn(dict(zip(batch.keys(), values)))
            for values in zip(*batch.values())
        ]

        if scalar_output:
            return {
                key: [example[key] for example in examples]
                for key in examples[0].keys()
            }
        else:
            return {
                key:
                list(itertools.chain(*(example[key] for example in examples)))
                for key in examples[0].keys()
            }
        return batch

    return execute_on_batch


def create_dataframe(dataset: datasets.DatasetDict):
    dataset.set_format("pandas")

    dataset_ = []
    for split, ds in dataset.items():
        split_df = ds[:]
        split_df["split"] = split
        dataset_.append(split_df)
    dataset_ = pd.concat(dataset_)
    dataset_.reset_index(drop=True, inplace=True)
    dataset.reset_format()

    return dataset_


def explode_qa(dataset: pd.DataFrame):
    dataset = dataset.explode(["questions", "answers"])
    dataset.rename(columns={
        "questions": "question",
        "answers": "answer"
    },
                   inplace=True)

    questions = pd.json_normalize(dataset["question"])
    questions = questions[["turn_id", "input_text"]]
    questions.rename(columns={
        "input_text": "question",
        "turn_id": "turn"
    },
                     inplace=True)

    answers = pd.json_normalize(dataset["answer"])
    answers = answers[[
        "input_text", "span_text", "span_start", "span_end", "answer_type"
    ]]
    answers.rename(columns={
        "input_text": "answer",
        "span_text": "rationale"
    },
                   inplace=True)

    dataset.reset_index(inplace=True)
    dataset.drop(["index", "question", "answer"], axis=1, inplace=True)
    dataset = dataset.join(questions)
    dataset = dataset.join(answers)

    cols = dataset.columns.tolist()
    cols.append(cols.pop(cols.index("last_turn")))
    cols.append(cols.pop(cols.index("qa_length")))
    cols.append(cols.pop(cols.index("split")))
    return dataset[cols]


def plot_answer_type_distribution(qa_dataset: pd.DataFrame):
    answer_type_distribution = qa_dataset.groupby(
        "split")["answer_type"].value_counts(normalize=True)
    answer_type_distribution = answer_type_distribution.apply(
        lambda x: np.round(x, decimals=3) * 100)
    answer_type_distribution = answer_type_distribution.rename(
        "frequency").reset_index()
    ax = sns.barplot(answer_type_distribution,
                     x="answer_type",
                     y="frequency",
                     hue="split")

    for i in ax.containers:
        ax.bar_label(i, )

    plt.tight_layout()
    plt.show()