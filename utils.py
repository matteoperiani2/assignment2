import os
import re
import string
import datasets

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from text_to_num import text2num


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


######################### Preprocessing #############################################
def remove_articles_(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)


def white_space_fix(text):
    return ' '.join(text.split())


def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)


def lower(text):
    return text.lower()


def normalize_text(text, remove_articles=False):
    """Lower text and remove punctuation, articles and extra whitespace."""
    text = remove_punc(lower(text))
    if remove_articles:
        text = remove_articles_(text)
    return white_space_fix(text)


def normalize_answer(text):
    """Lower text and remove punctuation, articles and extra whitespace."""
    return normalize_text(text, remove_articles=True)

def strip_non_alphanumeric_chars(text: str):
    """
    Removes trailing and leading non alpha-numeric characters from a given string.
    """
    start_index = 0
    while (start_index < len(text) and not text[start_index].isalnum()):
        start_index += 1

    end_index = len(text) - 1
    while (end_index > start_index and not text[end_index].isalnum()):
        end_index -= 1

    return text[start_index:end_index + 1]

def find_span(passage: str,
              text: str,
              span_start: int = None,
              span_end: int = None):

    if len(text) == 0: return (0, 0)
    assert text[0].isalnum() and text[-1].isalnum(), \
        "Text must begin and end with an alphanumeric character."

    start_idx = passage.find(text, span_start, span_end)
    end_idx = start_idx + len(text) - 1

    if start_idx == -1:
        raise ValueError("The text is not present in the passage.")

    # Find the beginning of the word in the passage
    while (start_idx > 0 and passage[start_idx - 1].isalnum()):
        start_idx -= 1

    # Find the end of the word in the passage
    while (end_idx < len(passage) - 1 and passage[end_idx + 1].isalnum()):
        end_idx += 1

    return start_idx, end_idx + 1

def fix_rationale(passage: str, rationale: str, span_start: int, span_end: int):
    rationale = strip_non_alphanumeric_chars(rationale)
    span_start, span_end = find_span(passage, rationale, span_start=span_start, span_end=span_end)
    return passage[span_start: span_end], span_start, span_end

####################################################################################################

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