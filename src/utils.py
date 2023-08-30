import inspect
import itertools
import os
import random
from typing import Dict, Literal

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from text_to_num import text2num



class AvgValue:
    def __init__(self, initial_value=0.0) -> None:
        self.__value = initial_value
        self.__last_value = initial_value
        self.__n = 0

    def update(self, value, n=1):
        self.__last_value = value
        old_n = self.__n
        self.__n += n
        self.__value = old_n / self.__n * self.__value + n / self.__n * value

    def value(self):
        return self.__value

    @property
    def last_value(self):
        return self.__last_value

    @property
    def n(self):
        return self.__n

class PropertyDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{key}'"
        )

    def __setattr__(self, key, value):
        self[key] = value

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

################# Reproducibility ######################à
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Using a generator and the following function as `worker_init_fn` preserves reproducibility when using DataLoader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_reproducible_dataloader(*args, **kwargs):
    generator = torch.Generator()
    return DataLoader(
        *args,
        **kwargs,
        #   worker_init_fn=seed_worker,
        #   generator=generator
    )


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
        text2num(string, "en", relaxed=True)
        return True
    except ValueError:
        return False


def batched_function(fn, scalar_output=True):
    def execute_on_batch(batch):
        examples = [
            fn(dict(zip(batch.keys(), values))) for values in zip(*batch.values())
        ]

        if scalar_output:
            return {
                key: [example[key] for example in examples]
                for key in examples[0].keys()
            }

        return {
            key: list(itertools.chain(*(example[key] for example in examples)))
            for key in examples[0].keys()
        }

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
    dataset.rename(columns={"questions": "question", "answers": "answer"}, inplace=True)

    questions = pd.json_normalize(dataset["question"])
    questions = questions[["turn_id", "input_text"]]
    questions.rename(
        columns={"input_text": "question", "turn_id": "turn"}, inplace=True
    )

    answers = pd.json_normalize(dataset["answer"])
    answers = answers[
        ["input_text", "span_text", "span_start", "span_end", "answer_type"]
    ]
    answers.rename(
        columns={"input_text": "answer", "span_text": "rationale"}, inplace=True
    )

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
    plot_distribution(qa_dataset, field="answer_type", hue="split")


def plot_distribution(dataset: pd.DataFrame, field: str, hue: str = None):
    if hue is not None:
        dataset = dataset.groupby(hue)

    distribution = dataset[field].value_counts(normalize=True)
    distribution = distribution.apply(lambda x: np.round(x, decimals=3) * 100)
    distribution = distribution.rename("frequency").reset_index()
    ax = sns.barplot(distribution, x=field, y="frequency", hue=hue)

    for i in ax.containers:
        ax.bar_label(
            i,
        )

    plt.tight_layout()
    plt.show()


def show_inputs(tokenizer, data, inputs):
    for k, v in inputs.items():
        print(f"{k:<27}: {v}")
    print()

    for idx in range(len(inputs["input_ids"])):
        show_input(tokenizer, data, inputs, idx)
        print()


def show_input(tokenizer, data, inputs, idx):
    sample_idx = (
        inputs["overflow_to_sample_mapping"][idx]
        if "overflow_to_sample_mapping" in inputs
        else idx
    )

    input_ids = np.asarray(inputs["input_ids"][idx])
    passage_mask = np.asarray(inputs["passage_mask"][idx])
    rationale_labels = np.asarray(inputs["rationale_labels"][idx])
    rationale_start = inputs["rationale_start"][idx]
    rationale_end = inputs["rationale_end"][idx]
    labels = np.asarray(inputs["labels"][idx])
    decoder_attention_mask = np.asarray(inputs["decoder_attention_mask"][idx])

    passage = input_ids[passage_mask.astype(np.bool_)]
    rationale = input_ids[rationale_labels > 0]
    assert np.all(rationale == input_ids[rationale_start:rationale_end])
    answer = labels[decoder_attention_mask.astype(np.bool_)]

    print("Input:", tokenizer.decode(input_ids))
    print("Q:", data["question"][sample_idx])
    print("P (-):", data["passage"][sample_idx])
    print("P (+):", tokenizer.decode(passage))
    print("R (-):", data["rationale"][sample_idx])
    print("R (+):", tokenizer.decode(rationale))
    print("A (-):", data["answer"][sample_idx])
    print("A (+):", tokenizer.decode(answer))
    print("History:", data["history"][sample_idx])


def logits_to_class(logits, task: Literal["binary", "multiclass"]) -> torch.LongTensor:
    if task == "binary":
        return (logits > 0.0).long()
    elif task == "multiclass":
        return torch.argmax(logits, dim=-1).long()
    else:
        raise ValueError(
            "Invalid task. Supported values are 'binary' and 'multiclass'."
        )


def prepare_model_inputs(
    model: nn.Module, inputs: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    forward_signature = set(inspect.signature(model.forward).parameters)
    inputs = {
        argument: value.to(model.device)
        for argument, value in inputs.items()
        if argument in forward_signature
    }
    return inputs
    
def print_conversation(passage, questions, answers, pred_answers, answers_f1, conv_f1):
    out = color.BOLD + "Passage: " + color.END + passage + "\n"
    out += "\n"
    for q,a,p_a,a_f1 in zip(questions, answers, pred_answers, answers_f1):
        out += color.BOLD + "Question: " + color.END + q + "\n"
        out += color.BOLD + "Predicted Answer: " + color.END + p_a + "\n"
        out += color.BOLD + "Answer: " + color.END + a + "\n"
        out += color.BOLD + "Answer SQUAD-f1: " + color.END + str(a_f1) + "\n"
        out += "\n"

    out += color.BOLD + "Conversation SQUAD-f1: " + color.END + str(conv_f1) + "\n"
    print(out)