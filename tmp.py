# %%
import os, urllib.request, inspect, functools, collections, gc
from tqdm.auto import tqdm
from typing import List, Optional, Union

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

import transformers, datasets
from transformers import AutoTokenizer, EncoderDecoderModel

import wandb

from src.generate_annotation import (
    annotate_dataset,
    create_readable_dataset,
    to_readable_story,
    AnswerType,
)
from src.utils import *
from src.preprocessing import *
from src.config import Config
from src.squad_f1 import squad_f1
from src.models import make_encoder_decoder_model

from src.pipeline import (
    make_teacher_force_scheduler,
    make_tokenizer,
    make_model,
    make_loss,
    make_optimizer,
    get_data,
    make_dataloader,
    make_scheduler,
    get_data,
    train,
    evaluate,
)

from src.evaluation import *

# keep datasets in memory if < 8 GB
datasets.config.IN_MEMORY_MAX_SIZE = 8 * 1024**3
CONFIG: Config = Config()

# # %% [markdown]
# # # [Task 1] Remove unaswerable QA pairs

# # %% [markdown]
# # ## Download the dataset

# # %%
# class DownloadProgressBar(tqdm):

#     def update_to(self, b=1, bsize=1, tsize=None):
#         if tsize is not None:
#             self.total = tsize
#         self.update(b * bsize - self.n)


# def download_url(url, output_path):
#     with DownloadProgressBar(unit='B',
#                              unit_scale=True,
#                              miniters=1,
#                              desc=url.split('/')[-1]) as t:
#         urllib.request.urlretrieve(url,
#                                    filename=output_path,
#                                    reporthook=t.update_to)


# def download_data(url_path, output_path, split):
#     create_dirs_for_file(output_path)

#     if not os.path.exists(output_path):
#         print(f"Downloading CoQA {split} data split... (it may take a while)")
#         download_url(url=url_path, output_path=output_path)
#         print("Download completed!")

# # %%
# download_data(CONFIG.dataset.train_url, CONFIG.dataset.train_data_raw, "train")
# download_data(CONFIG.dataset.test_url, CONFIG.dataset.test_data_raw, "test")

# # %% [markdown]
# # ## Remove unanswerable QA pairs

# # %% [markdown]
# # We perform an automated annotation process, where we assign each answer a specific answer type:
# # - `unknown`: unanswerable question - we check if (`span_start`, `span_end`) = (-1, -1)
# # - `span`: answer overlaps with the passage, after ignoring punctuation and case mismatches
# # - `yes_no`: answer is Yes or No
# # - `counting`: answer is the result of a counting process - we check if the answer is a written number or a digit
# # - `multiple_choice`: answer is one the choices provided in the question - we check if the question has an `or` and contains the answer
# # - `fluency`: changes to the text span to improve fluency - if the answer is none of the previous, we flag it as `fluency`
# # 
# # Although the automated annotation procedure is susceptible to errors, with some `span` answers being annotated as `counting`, we believe that it is reliable and provides a useful starting point for analyzing the dataset.

# # %%
# def generate_annotated_dataset(raw_filename,
#                                annotated_filename,
#                                ignore_additional_answers=True):
#     if not os.path.exists(annotated_filename):
#         print("Generating annotated data...", end="")
#         annotate_dataset(raw_filename,
#                          annotated_filename,
#                          ignore_additional_answers=ignore_additional_answers)
#         print("Done!")


# generate_annotated_dataset(CONFIG.dataset.train_data_raw,
#                            CONFIG.dataset.train_data_annotated)

# generate_annotated_dataset(CONFIG.dataset.test_data_raw,
#                            CONFIG.dataset.test_data_annotated)

# # %%
# target = datasets.load_dataset("json",
#                              data_files=CONFIG.dataset.train_data_annotated,
#                              field="data")
# test_data = datasets.load_dataset(
#     "json",
#     data_files=CONFIG.dataset.test_data_annotated,
#     field="data",
#     split="train")
# target["test"] = test_data.remove_columns(
#     list(set(test_data.features).difference(target["train"].features)))

# print(target)

# # %% [markdown]
# # Let's remove unknown qa pairs.

# # %%
# def remove_unknown_qa(example):
#     questions = []
#     answers = []
#     last_turn = 0
#     for question_item, answer_item in zip(example["questions"],
#                                           example["answers"]):
#         last_turn += 1
#         if question_item["answer_type"] != "unknown":
#             questions.append(question_item)
#             answers.append(answer_item)

#     qa_length = len(questions)
#     return {
#         'questions': questions,
#         'answers': answers,
#         'qa_length': qa_length,
#         'last_turn': last_turn,
#     }


# print("Removing unknown qa pairs...")
# filtered_dataset = target.map(remove_unknown_qa)
# print("Done!")

# print(filtered_dataset)

# # %% [markdown]
# # We have to check whether there are examples with no qa pairs and remove them from the dataset.

# # %%
# def remove_examples_with_no_qa(dataset, verbose=True):
#     if verbose:
#         examples_with_no_qa = dataset.filter(
#             lambda example: example["qa_length"] == 0)
#         print("Examples with no qa pairs:", examples_with_no_qa.num_rows)

#         examples_with_no_qa = examples_with_no_qa["train"]
#         examples_with_no_qa.set_format("pandas")
#         display(examples_with_no_qa[:])

#         print()
#         print("Filtering out examples with no qa pairs...")
#     filtered_dataset = dataset.filter(
#         lambda example: example["qa_length"] > 0)
#     if verbose:
#         print("Done!")

#         print()
#         print("Number of examples:", filtered_dataset.num_rows)

#     return filtered_dataset

# # %%
# filtered_dataset = remove_examples_with_no_qa(filtered_dataset)

# # %%
# filtered_dataset.save_to_disk(CONFIG.dataset.filtered_dir)
# del filtered_dataset
# del target

# # %% [markdown]
# # ## Data Inspection

# # %%
# target = datasets.load_from_disk(CONFIG.dataset.filtered_dir)
# print(target)

# # %% [markdown]
# # The dataset is very difficult to explore in a Jupyter notebook. To overcome this, we create a readable `.txt` version, similar to the one provided by the authors of CoQA. For each story, the format is:
# # 
# # ```
# # source: <source>, id: <id>
# # 
# # <passage>
# # 
# # turn: 1
# # Q   <question_1> || <question_type_1>
# # A   <answer_1> || <rationale_1> || <answer_type_1>
# #                         .
# #                         .
# #                         .
# # turn: i
# # Q   <question_i> || <question_type_i>
# # A   <answer_i> || <rationale_i> || <answer_type_i>
# #                         .
# #                         .
# #                         .
# # ```

# # %%
# def generate_readable_dataset(annotated_filename,
#                               readable_filename,
#                               ignore_additional_answers=True):
#     if not os.path.exists(readable_filename):
#         print("Generating readable data...", end="")
#         create_readable_dataset(
#             annotated_filename,
#             readable_filename,
#             ignore_additional_answers=ignore_additional_answers)
#         print("Done!")


# generate_readable_dataset(CONFIG.dataset.train_data_annotated,
#                           CONFIG.dataset.train_data_readable)
# generate_readable_dataset(CONFIG.dataset.test_data_annotated,
#                           CONFIG.dataset.test_data_readable)

# # %% [markdown]
# # Here, we show an example of story.

# # %%
# sample_id = np.random.choice(target["train"].num_rows)
# sample = target["train"][sample_id]

# story_signature = set(inspect.signature(to_readable_story).parameters)
# story_kwargs = {
#     argument: value
#     for argument, value in sample.items() if argument in story_signature
# }
# readable_story = to_readable_story(**story_kwargs)
# print(readable_story)

# # %% [markdown]
# # Let's see some statistics.

# # %%
# df = create_dataframe(target)
# del target

# print(f"Number of passages: {len(df)}")
# print(f"Number of QA pairs: {df['qa_length'].sum()}")
# print(f"Number of files: {len(df['filename'].unique())}")

# # %%
# print(f"Questions dictionary keys:\t {list(df.loc[0, 'questions'][0].keys())}")
# print(f"Answers dictionary keys:\t {list(df.loc[0, 'answers'][0].keys())}")

# # %%
# print("Distribution of splits: ")
# split_counts = df['split'].value_counts()
# print(split_counts)

# # %%
# print("Conversations with gaps in the history (%)")
# broken_hist_counts = df[
#     df["qa_length"] < df["last_turn"]]["split"].value_counts()
# print(broken_hist_counts / split_counts * 100)

# # %%
# display(df["qa_length"].describe())
# plt.title("Conversation length")
# sns.boxplot(data=df, x="qa_length", y="split", showmeans=True);

# # %% [markdown]
# # Here, we want to statistics about number of words/tokens.

# # %%
# tokenizers_ = {
#     k: AutoTokenizer.from_pretrained(checkpoint).tokenize
#     for k, checkpoint in CONFIG.checkpoints.__dict__.items()
# }
# tokenizers = {"": str.split}
# tokenizers.update(tokenizers_)

# # %%
# def plot_length(dataset: pd.DataFrame,
#                 column: str,
#                 column_name: Optional[str] = None,
#                 split_fn=str.split,
#                 notes: Optional[str] = None,
#                 max_length=None):
#     if column_name is None:
#         column_name = column

#     length_col = column_name + "_length"
#     length = dataset[column].apply(split_fn).apply(len)
#     dp = dataset[["split"]].copy()
#     dp[length_col] = length

#     if notes:
#         print(notes)
#     display(dp.groupby("split").describe())

#     fig, axes = plt.subplots(1, 2, figsize=(15, 5))
#     boxplot = sns.boxplot(data=dp, x=length_col, y="split", showmeans=True, ax=axes[0])
#     histplot = sns.histplot(dp,
#                  x=length_col,
#                  hue="split",
#                  stat="density",
#                  common_norm=False,
#                  discrete=True,
#                  ax=axes[1])
#     if max_length is not None:
#         boxplot.axvline(x=max_length, color='red', ls='--')
#         histplot.axvline(x=max_length, color='red', ls='--')
#     fig.suptitle(f"{column_name.capitalize()} length ({notes})", fontsize=16)
#     plt.show()

# # %%
# for checkpoint_name, tokenize_fn in tokenizers.items():
#     plot_length(df,
#                 column="story",
#                 column_name="passage",
#                 split_fn=tokenize_fn,
#                 notes=checkpoint_name,
#                 max_length=512)

# # %% [markdown]
# # To analyze questions, answers and rationales, let's explode them.

# # %%
# qa_dataset = explode_qa(df)
# assert not np.any(qa_dataset["answer_type"] == "unknown")
# del df

# # %%
# for checkpoint_name, tokenize_fn in tokenizers.items():
#     plot_length(qa_dataset,
#                 column="question",
#                 split_fn=tokenize_fn,
#                 notes=checkpoint_name)

# # %%
# for checkpoint_name, tokenize_fn in tokenizers.items():
#     plot_length(qa_dataset,
#                 column="answer",
#                 split_fn=tokenize_fn,
#                 notes=checkpoint_name)

# # %% [markdown]
# # There are some very long answers. It's likely most of them are trash and can be removed from the dataset.

# # %% [markdown]
# # Let's inspect the answer length per answer_type.

# # %%
# def plot_answer_length(dataset, split_fn=str.split, notes=None):
#     length_col = "answer" + "_length"
#     length = dataset["answer"].apply(split_fn).apply(len)
#     dp = dataset[["answer_type", "split"]].copy()
#     dp[length_col] = length

#     if notes:
#         print(notes)
#     display(dp.groupby(["answer_type", "split"]).describe())

#     answer_types = AnswerType.list(return_unknown=False)
#     fig = plt.figure(figsize=(15, 4 * len(answer_types)))
#     fig.suptitle(f"Answer length ({notes})", fontsize=16)

#     subfigs = fig.subfigures(nrows=len(answer_types), ncols=1)
#     for answer_type, subfig in zip(answer_types, subfigs):
#         data = dp[dp["answer_type"] == answer_type]
#         subfig.suptitle(answer_type)
#         axes = subfig.subplots(1, 2)
#         sns.boxplot(data=data,
#                     x=length_col,
#                     y="split",
#                     showmeans=True,
#                     ax=axes[0])
#         sns.histplot(data,
#                      x=length_col,
#                      hue="split",
#                      stat="density",
#                      common_norm=False,
#                      discrete=True,
#                      ax=axes[1])
#     plt.show()
    
# for checkpoint_name, tokenize_fn in tokenizers.items():
#     plot_answer_length(qa_dataset, split_fn=tokenize_fn, notes=checkpoint_name)

# # %% [markdown]
# # It's likely that very long answers need to be cleaned or to be removed. Let's start with `SPAN` answers.

# # %%
# def to_readable_question(question, answer, rationale, answer_type, turn=None):
#     lines = []
#     if turn is not None:
#         lines.append(f'turn: {turn}')
#     lines.append(f'Q\t\t{question} || {answer_type}')
#     lines.append(f'A\t\t{answer} || {rationale}')

#     return '\n'.join(lines)


# def print_readable_questions(data: pd.DataFrame, show_turn=False):
#     for _, row in data.iterrows():
#         print("id:", row["id"])
#         turn = row["turn"] if show_turn else None
#         print(
#             to_readable_question(row["question"],
#                                  row["answer"],
#                                  row["rationale"],
#                                  row["answer_type"],
#                                  turn=turn))
#         print()

# def show_long_answers(qa_dataset, answer_type, length):
#     answers = qa_dataset[qa_dataset["answer_type"] == str(answer_type)].copy()
#     answers["length"] = answers["answer"].str.split().apply(len)
#     answers = answers.sort_values("length", ascending=False)
#     long_answers = answers[answers["length"] > length]
#     print(f"{answer_type} answers longer than {length} words:", len(long_answers))
#     print_readable_questions(long_answers)

# # %% [markdown]
# # By manual inspection, we found that all the `span` answers longer than 37 words are useless, since they are just parts of the passage and do not answer their relative question.

# # %%
# show_long_answers(qa_dataset, answer_type = AnswerType.SPAN, length = 37)

# # %% [markdown]
# # `Yes_no` answers with more than 1 words need to be cleaned.

# # %%
# show_long_answers(qa_dataset, answer_type = AnswerType.YES_NO, length = 1)

# # %% [markdown]
# # `Counting` answers with two words are ok. The examples below show also that some of the answers annotated as `counting` are rather `fluency`.

# # %%
# show_long_answers(qa_dataset, answer_type = AnswerType.COUNTING, length = 1)

# # %% [markdown]
# # Also `multiple_choice` answers with more than 1 word are ok.

# # %%
# show_long_answers(qa_dataset, answer_type = AnswerType.MULTIPLE_CHOICE, length = 1)

# # %% [markdown]
# # Let's inspect the question-passage length, i.e. the input length when there is no history.

# # %%
# qa_dataset["input"] = qa_dataset["question"] + " " + qa_dataset["story"]
# for checkpoint_name, tokenize_fn in tokenizers.items():
#     plot_length(qa_dataset,
#                 column="input",
#                 column_name="question-passage",
#                 split_fn=tokenize_fn,
#                 notes=checkpoint_name,
#                 max_length=512)

# # %% [markdown]
# # Let's analyze the question-answer pair lengths, useful to have information on the history length.

# # %%
# qa_dataset["qa"] = qa_dataset["question"] + " " + qa_dataset["answer"]
# for checkpoint_name, tokenize_fn in tokenizers.items():
#     plot_length(qa_dataset,
#                 column="qa",
#                 column_name="question-answer pair",
#                 split_fn=tokenize_fn,
#                 notes=checkpoint_name)

# # %%
# for checkpoint_name, tokenize_fn in tokenizers.items():
#     plot_length(qa_dataset,
#                 column="rationale",
#                 split_fn=tokenize_fn,
#                 notes=checkpoint_name)

# # %% [markdown]
# # Most of the very lengthy rationales are a consequence of poor workmanship.

# # %%
# def show_long_rationales(qa_dataset, length):
#     rationales = qa_dataset.copy()
#     rationales["length"] = rationales["rationale"].str.split().apply(len)
#     rationales = rationales.sort_values("length", ascending=False)
#     long_rationales = rationales[rationales["length"] > length]
#     print("Number of questions with very long rationale:", len(long_rationales))
#     print()
#     print_readable_questions(long_rationales)
    
# show_long_rationales(qa_dataset, length=150)

# # %%
# example = qa_dataset[qa_dataset["rationale"] == qa_dataset["story"]]
# print("Number of questions with rationale equal to the whole passage:",
#       len(example))
# print()
# print_readable_questions(example)

# # %% [markdown]
# # Upon further analysis of the corpus, we observed instances where words at the beginning or at the end of the rationale are truncated. Therefore, we have to fix them if we intend to use the rationale to train the network.

# # %%
# def print_rationales(data: pd.DataFrame):
#     for i, row in data.iterrows():
#         print("id:", row["id"], "split:", row["split"])
#         print("turn:", row["turn"])
#         print("R:", row["rationale"], "|", row["span_start"], "-",
#               row["span_end"])
#         print("=" * 100)


# def is_rationale_truncated(passage, span_start, span_end):
#     if span_start > 0 and passage[span_start].isalnum() and passage[
#             span_start - 1].isalnum():
#         return True

#     if span_end < len(passage) and passage[
#             span_end - 1].isalnum() and passage[span_end].isalnum():
#         return True

#     return False


# example = qa_dataset[qa_dataset.apply(lambda row: is_rationale_truncated(
#     row["story"], row["span_start"], row["span_end"]),
#                                       axis=1)]
# print("Number of rationales with errors:", len(example), "/", len(qa_dataset))
# print()
# print_rationales(example)

# # %% [markdown]
# # In order to solve the issue, we have to:
# # 1. remove any leading or trailing non alphanumeric character (i.e. spaces, punctuation, etc.) from the rationale.
# # 2. compute new `span_start` and `span_end` indices by considering the entire words.

# # %%
# #1. removing leading and trailing non alphanumeric chars
# example = " Despite a common background, the groups' views on religious toleration were mixe."
# print("Before:", example)
# print("After:", strip_non_alphanumeric_chars(example))

# # %%
# qa_dataset_fixed = qa_dataset.copy()
# qa_dataset_fixed["rationale"] = qa_dataset_fixed["rationale"].apply(
#     strip_non_alphanumeric_chars)
# n_affected_rationales = np.count_nonzero(
#     qa_dataset_fixed["rationale"] != qa_dataset["rationale"])
# print("Number of affected rationales:", n_affected_rationales, "/",
#       len(qa_dataset))


# # %%
# #2. compute span_start and span_end indices by considering the entire words.
# passage = "\tThe Vatican Library is a research library for history, law, philosophy, science and theology."
# rationale = "e Vatican Library is a research lib"
# span_start, span_end = find_span(passage, rationale)
# fixed_rationale = passage[span_start:span_end]
# assert fixed_rationale == "The Vatican Library is a research library"

# print("P:", passage)
# print("R (-):", rationale)
# print("R (+):", fixed_rationale, "|", span_start, "-", span_end)

# print()

# passage = " Mandi Marie Utash posted Friday to a GoFundMe.com page [...]\nMandi Marie Utash\twrote that her father doesn't seem to know what happened"
# rationale = "andi Marie Utas"
# span_start, span_end = find_span(passage,
#                                  rationale,
#                                  span_start=63,
#                                  span_end=78)
# fixed_rationale = passage[span_start:span_end]
# assert fixed_rationale == "Mandi Marie Utash"

# print("P:", passage)
# print("R (-):", rationale)
# print("R (+):", fixed_rationale, "|", span_start, "-", span_end)

# # %%
# # Fix rationale by applying (1) and (2)
# passage = "\tThe Vatican Library is a research library for history, law, philosophy, science and theology."
# rationale = " e Vatican Library is a research library for history,  "
# fixed_rationale, span_start, span_end = fix_rationale(passage, rationale, 1,
#                                                       100)

# print("P:", passage)
# print("R (-):", rationale)
# print("R (+):", fixed_rationale, "|", span_start, "-", span_end)
# assert fixed_rationale == "The Vatican Library is a research library for history"

# # %%
# def _fix_rationale(row):
#     fixed_rationale, span_start, span_end = fix_rationale(
#         row["story"], row["rationale"], row["span_start"], row["span_end"])

#     row["rationale"] = fixed_rationale
#     row["span_start"] = span_start
#     row["span_end"] = span_end

#     return row


# qa_dataset_fixed = qa_dataset.apply(_fix_rationale, axis=1)
# diff_start = qa_dataset_fixed["span_start"] != qa_dataset["span_start"]
# diff_end = qa_dataset_fixed["span_end"] != qa_dataset["span_end"]
# diff_rationale = qa_dataset_fixed["rationale"] != qa_dataset["rationale"]

# assert np.all((diff_start | diff_end) == diff_rationale)

# example = qa_dataset_fixed[diff_rationale]
# print("Number of fixed rationales:", len(example), "/", len(qa_dataset))
# print()
# for i, row in example.iloc[:100].iterrows():
#     old_row = qa_dataset[(qa_dataset["id"] == row["id"])
#                          & (qa_dataset["turn"] == row["turn"])]
#     assert len(old_row) == 1
#     old_row = old_row.iloc[0]

#     print("id:", row["id"], "split:", row["split"])
#     print("turn:", row["turn"])
#     print("R (-):", old_row["rationale"], "|", old_row["span_start"], "-",
#           old_row["span_end"])
#     print("R (+):", row["rationale"], "|", row["span_start"], "-",
#           row["span_end"])
#     print("=" * 100)

# qa_dataset = qa_dataset_fixed
# del qa_dataset_fixed

# # %% [markdown]
# # The type of answers are equally distributed across datasets.

# # %%
# plot_answer_type_distribution(qa_dataset)

# # %%
# for answer_type in AnswerType.list(return_unknown=False):
#     example = qa_dataset[qa_dataset["answer_type"] == answer_type].sample(3)
#     print(answer_type)
#     print()
#     print_readable_questions(example)
#     print("=" * 100)

# # %% [markdown]
# # Yes and no answers are almost equally distributed.

# # %%
# def plot_yes_no_distribution(qa_dataset: pd.DataFrame):
#     yes_no_answers = qa_dataset[qa_dataset["answer_type"] == "yes_no"].copy()
#     yes_no_answers["answer"] = yes_no_answers["answer"].apply(normalize_answer)
#     plot_distribution(yes_no_answers, field="answer", hue="split")

# plot_yes_no_distribution(qa_dataset)

# # %%
# del qa_dataset

# # %% [markdown]
# # ## Remove long answers from training set

# # %% [markdown]
# # In the Data Inspection section we found that long `span` answers are simply wrong. Therefore, we remove it from the training set before going on.

# # %%
# dataset = datasets.load_from_disk(CONFIG.dataset.filtered_dir)
# dataset

# # %%
# def remove_long_span_answers(example, max_length):
#     questions = []
#     answers = []
#     for question_item, answer_item in zip(example["questions"],
#                                           example["answers"]):
        
#         if question_item["answer_type"] == str(AnswerType.SPAN):
#             answer = answer_item["input_text"]
#             length = len(answer.split())
#             if length > max_length:
#                 continue

#         questions.append(question_item)
#         answers.append(answer_item)

#     qa_length = len(questions)
#     return {
#         'questions': questions,
#         'answers': answers,
#         'qa_length': qa_length
#     }


# print("Removing long span answers...")
# dataset["train"] = dataset["train"].map(remove_long_span_answers, fn_kwargs={"max_length": CONFIG.span_max_length})
# print("Done!")

# filtered_dataset = remove_examples_with_no_qa(dataset)
# filtered_dataset.save_to_disk(CONFIG.dataset.filtered_dir)

# del dataset
# del filtered_dataset

# # %% [markdown]
# # ## Input preparation

# # %% [markdown]
# # In order to train our models, we have to prepare the inputs properly. First of all, we explode the QA pairs so that to have one record for each question. Then, we preprocess the texts. In particular:
# # 1. we fix the rationales by removing leading and trailing non alphanumeric chars, and by restoring truncated words
# # 2. we remove multiple spaces from the texts of our inputs (changing the rationale spans accordingly)
# # 3. we clean the answers by removing leading and trailing non alphanumeric chars

# # %%
# dataset = datasets.DatasetDict.load_from_disk(CONFIG.dataset.filtered_dir)
# train_dataset = dataset["train"].select(range(10))

# preprocessing = CoQADatasetPreprocessing()
# train_dataset = train_dataset.map(
#     batched_function(preprocessing.explode_questions, scalar_output=False),
#     batched=True,
#     remove_columns=dataset["train"].column_names,
# )
# train_dataset = train_dataset.rename_column("story", "passage")
# train_dataset = train_dataset.remove_columns(["questions", "answers", "qa_length", "last_turn"])
# train_dataset = train_dataset.map(
#     batched_function(preprocessing.preprocess_texts), batched=True
# )

# print(train_dataset)
# train_dataset[:5]

# # %% [markdown]
# # ### No History

# # %% [markdown]
# # Given a passage $P$ and a question $Q$, the corresponding input will be obtained by concatenating $P$ to $Q$ separated by a model-dependant separator.
# # - DistilRoBERTa input: `<s>Q</s></s>P</s><pad>...<pad>`
# # - BERTTiny input: `[CLS] Q [SEP] P [SEP] [PAD] ... [PAD]`
# # 
# # Here, we have to address two problems:
# # - the input may exceed the `max_sequence_length` of the model (i.e. 512 tokens). In such case, we truncate only the passage $P$.
# # - if we want to use the rationale, we need to map the rational span chars indices to the corresponding tokens indices.
# # 
# # Truncating the passage leads to another problem: the rationale (hence, the answer) may not be contained in the truncated passage. A way to handle such problem is by considering several "windows" of the passage for each QA pair.

# # %%
# def process_data_to_model_inputs(tokenizer, data, add_history=False):
#     preprocessing = CoQADatasetPreprocessing(tokenizer,
#                                              use_window=True,
#                                              encoder_max_length=300)
#     return preprocessing.process_data_to_model_inputs(data,
#                                                       add_history=add_history)


# def prepare_and_show_inputs(checkpoint, data, add_history=False):
#     tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#     inputs = process_data_to_model_inputs(tokenizer,
#                                           data,
#                                           add_history=add_history)
#     show_inputs(tokenizer, data, inputs)

# # %%
# indices = np.random.choice(train_dataset.num_rows, size=3)
# target = train_dataset.select(indices)[:]

# for checkpoint_name, checkpoint in CONFIG.checkpoints.__dict__.items():
#     print("Input examples for:", checkpoint_name)
#     print()
#     prepare_and_show_inputs(checkpoint, target)
#     print("=" * 30)

# # %% [markdown]
# # ### History

# # %% [markdown]
# # Given a passage $P$, a question $Q_i$ and an history $H=\{(Q_j, A_j)\}_{j=1..i-1}$, the corresponding input will be obtained as the concatenation of $H$, $Q$ and $P$ separated by a model-dependant separator, as follows:
# # - DistilRoBERTa input: `<s>Q1</s>A1</s>...Qi-1</s>Ai-1</s>Qi</s></s></s>P</s><pad>...<pad>`
# # - BERTTiny input: `[CLS] Q1 [SEP] A1 [SEP] ... Qi-1 [SEP] Ai-1 [SEP] Qi [SEP] P [SEP] [PAD] ... [PAD]`
# # 
# # By adding the history, the input very often exceed the `max_sequence_length` of the model (i.e. 512 tokens). From the analysis above, we know that each QA pair requires around 10 tokens. Hence, an history of $[3..5]$ passes is a good trade-off between the context and the needed space.

# # %%
# indices = np.random.choice(train_dataset.num_rows, size=3)
# target = train_dataset.select(indices)[:]
# for checkpoint_name, checkpoint in CONFIG.checkpoints.__dict__.items():
#     print("Input examples for:", checkpoint_name)
#     print()
#     prepare_and_show_inputs(checkpoint, target, add_history=True)
#     print("=" * 30)

# # %%
# del dataset
# del train_dataset
# del target

# # %% [markdown]
# # # [Task 2] Train, test and validation split

# # %%
# dataset = datasets.load_from_disk(CONFIG.dataset.filtered_dir)
# print(dataset)

# # %%
# train_data = dataset["train"]
# test_data = dataset["test"]
# splitted_dataset = train_data.train_test_split(train_size=0.8, seed=42)
# splitted_dataset["validation"] = splitted_dataset.pop("test")
# splitted_dataset["test"] = test_data

# print(splitted_dataset)

# splitted_dataset.save_to_disk(CONFIG.dataset.splitted_dir)
# del train_data
# del test_data

# # %% [markdown]
# # Before proceeding with the next task, we explode the questions and apply the preprocessing as discussed in the Input Preparation section of the first task.

# # %%
# preprocessing = CoQADatasetPreprocessing(**CONFIG.preprocessing.__dict__)
# dataset = splitted_dataset.map(batched_function(preprocessing.explode_questions, scalar_output=False), batched=True)
# dataset = dataset.rename_column("story", "passage")
# dataset = dataset.remove_columns(["questions", "answers", "qa_length", "last_turn"])
# dataset = dataset.map(batched_function(preprocessing.preprocess_texts), batched=True)

# dataset.save_to_disk(CONFIG.dataset.processed_dir)

# print(dataset)
# dataset["train"][:5]

# # %% [markdown]
# # The different types of answer are equally distributed across the three splits.

# # %%
# qa_dataset = create_dataframe(dataset)
# plot_answer_type_distribution(qa_dataset)

# # %%
# del qa_dataset
# del splitted_dataset
# del dataset

# # %% [markdown]
# # # [Task 3] Model definition

# # %% [markdown]
# # We use an hybrid approach, i.e. both extractive and generative. We use an encoder-decoder model, where the encoder also predicts if the answer is yes/no/generative and the probability that each token is in the rationale. The last hidden state of the passage are multiplied by this probability before being passed to the decoder model.
# # 
# # All models implementation are in the file *src/models.py*

# # %% [markdown]
# # # [TASK 4] Question generation with text passage $P$, question $Q$

# # %% [markdown]
# # We want to define $f_\theta(P, Q)$: consider a dialogue on text passage $P$. For each question $Q_i$ at dialogue turn $i$, we want $f_\theta(P, Q_i) \approx A_i$.
# # 
# # - Our question-answering model requires two inputs: the passage $P$ and the question $Q_i$.
# # - The model generates the answer $A_i$.
# # 
# # 
# # To facilitate easy access to these inputs and outputs, we need to prepare our inputs properly. As discussed in [Task 1], given a passage $P$ and a question $Q$, the corresponding input will be obtained by concatenating $P$ to $Q$ separated by a model-dependant separator.
# # - DistilRoBERTa input: `<s>Question</s></s>Passage</s><pad>...<pad>`
# # - BERTTiny input: `[CLS] Question [SEP] Passage [SEP] [PAD] ... [PAD]`

# # %% [markdown]
# # The `generate_answer` method takes as input the passage(s) and the question(s) and generate the corresponding answer(s). Of course, since our models are not trained, the output is meaningless.

# # %%
# class QuestionAnswerer():
#     def __init__(self,
#                  tokenizer,
#                  model,
#                  preprocessing_config: dict = {},
#                  device=None) -> None:
#         self.preprocessing_config = preprocessing_config
#         self.tokenizer = tokenizer
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = model.to(self.device)
#         self.model.eval()
#         self.preprocessing = CoQADatasetPreprocessing(self.tokenizer, **preprocessing_config)

#     def generate_answer(
#             self,
#             passage: Union[str, List[str]],
#             question: Union[str, List[str]],
#             history: Optional[Union[str, List[str]]] = None
#     ) -> List[str]:
#         use_history = history is not None
#         preprocess = batched_function(self.preprocessing.preprocess_texts)
        
#         if isinstance(passage, str):
#             passage = [passage]
#             question = [question]
#             history = [history] if use_history else []

#         inputs = {
#             "id": list(range(len(passage))),
#             "passage": passage,
#             "question": question,
#         }

#         if use_history:
#             inputs["history"] = history

#         inputs = preprocess(inputs)
#         inputs = self.preprocessing.process_data_to_model_inputs(
#             inputs, add_history=use_history, padding="max_length")
#         inputs = inputs.convert_to_tensors("pt")

#         return self.generate_answer_from_input_tensors(inputs)

#     def generate_answer_from_input_tensors(self, inputs):
#         forward_signature = set(
#             inspect.signature(self.model.forward).parameters)
#         forward_signature.remove("decoder_input_ids")
#         inputs = {
#             argument: value.to(self.device)
#             for argument, value in inputs.items()
#             if argument in forward_signature
#         }

#         outputs = self.model.generate(**inputs)
#         output_str = self.tokenizer.batch_decode(outputs, skip_special_token=True)
#         return output_str


# # %% [markdown]
# # # [Task 5] Question generation with text passage $P$, question $Q$ and dialogue history $H$

# # %% [markdown]
# # We want to define $f_\theta(P, Q, H)$: consider a dialogue on text passage $P$. For each question $Q_i$ at dialogue turn $i$ and the corresponding dialogue history $H_i = \{ Q_0, A_0, \dots, Q_{i-1}, A_{i-1} \}$, we want $f_\theta(P, Q_i, H_i) \approx A_i$.
# # 
# # - Our question-answering model requires three inputs: the passage $P$, the question $Q_i$ and, the history $H_i$.
# # - The model generates the answer $A_i$.
# # 
# # To facilitate easy access to these inputs and outputs, we need to prepare our inputs properly. As discussed in [Task 1], truncating the history from the end seems reasonable. Hence given a passage $P$, a question $Q$ and an history $H=\{ Q_{i-k}, A_{i-k}, \dots, Q_{i-1}, A_{i-1} \}$ at turn $i$, the corresponding input will be obtained as the concatenation of $H$, $Q$ and $P$ separated by a model-dependant separator, as follows:
# # - DistilRoBERTa input: `<s>Qi-k</s>Ai-k</s>...Qi-1</s>Ai-1</s>Qi</s></s></s>P</s><pad>...<pad>`
# # - BERTTiny input: `[CLS] Qi-k [SEP] Ai-k [SEP] ... Qi-1 [SEP] Ai-1 [SEP] Qi [SEP] P [SEP] [PAD] ... [PAD]`
# # 
# # We choose $k=4$.

# # %% [markdown]
# # To generate the answer as explained above, we exploit the method `generate_answer` of the class `QuestionAnswerer` defined in the task before. Indeed, it takes an additional parameter `history`.

# # %% [markdown]
# # # [Task 6] Train and evaluate $f_\theta(P, Q)$ and $f_\theta(P, Q, H)$

# # %% [markdown]
# # First of all, we have to process data to get inputs for the two models. Here, we have to understand how to handle questions whose number of tokens does not fit in the two models (i.e. number of tokens > 512), in the training phase.
# # We have two options:
# # - truncate the passage and discard the question if the rationale is not contained in the input. If the rationale is not in the passage, the question is likely to be unanswerable.
# # - take multiple windows of the same passage and sample the ones which contain the rationale.
# # 
# # The first approach is far easier and seems to be reasonable given that we have to discard very few questions.

# # %% [markdown]
# # ## Create train dataset

# # %%
# for checkpoint_name, checkpoint in CONFIG.checkpoints.__dict__.items():
#     print("Processing data for:", checkpoint_name)
#     dataset = datasets.load_from_disk(CONFIG.dataset.processed_dir)
#     tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#     preprocessing = CoQADatasetPreprocessing(tokenizer, **CONFIG.preprocessing.__dict__)
#     dataset = dataset.map(
#         preprocessing.process_data_to_model_inputs,
#         fn_kwargs={"add_history": False},
#         batched=True,
#         remove_columns=dataset["train"].column_names,
#         num_proc=None,
#     )
#     num_rows = dataset["train"].num_rows
#     print("Number of rows in the training set before removing potentially unanswerable questions:", num_rows)
#     dataset["train"] = dataset["train"].filter(lambda example: np.asarray(example["rationale_start"]) != -1, batched=True)
#     print("Number of rows in the training set after removing potentially unanswerable questions:", dataset["train"].num_rows)
#     removed_rows = num_rows - dataset["train"].num_rows
#     removed_ratio = removed_rows / num_rows
#     print(f"Removed rows: {removed_rows} ({removed_ratio:.2%})")
#     dataset.save_to_disk(CONFIG.dataset.train_no_history(checkpoint_name))
# dataset

# # %% [markdown]
# # ## Train

# # %%
# hyperparameters = PropertyDict(
#     seed=42,
#     checkpoint_name="bert_tiny",
#     model_name="bert_tiny",
#     model_type="encoder_decoder",
#     initialize_cross_attention=True,
#     yng_loss_weight=0.6,
#     rationale_loss_weight=0.8,
#     generative_loss_weight=0.2,
#     batch_size=32,
#     val_batch_size=64,
#     generate_batch_size=32,
#     num_workers=2,
#     num_epochs=3,
#     optimizer_name="AdamW",
#     learning_rate=2e-4,
#     scheduler="linear",
#     warmup_fraction=0.1,
#     teacher_force_scheduler="linear",
#     tf_start = 1.,
#     tf_end = 0.,
#     tf_fraction = 0.6,
#     accumulation_steps=1,
#     gradient_clip=1.0,
#     mixed_precision="fp16",
#     checkpoint_interval=700,
#     log_interval=700,
#     cpu=False,
# )

# with wandb.init(project=CONFIG.wandbConfig.project, config=hyperparameters):
#     val_conifg = wandb.config

#     set_seed(val_conifg.seed)

#     # Make the model
#     tokenizer = make_tokenizer(val_conifg)
#     model = make_model(val_conifg, tokenizer)

#     # Make the data
#     train_data = get_data("train", val_conifg)
#     val_data = get_data("validation", val_conifg)
#     train_dataloader = make_dataloader(train_data, tokenizer, val_conifg, split="train")
#     val_dataloader = make_dataloader(val_data, tokenizer, val_conifg, split="validation")

#     # Make the loss, the optimizer and the scheduler
#     loss_fn = make_loss(val_conifg)
#     optimizer = make_optimizer(model, loss_fn, val_conifg)
#     scheduler = make_scheduler(
#         optimizer, steps_per_epoch=len(train_dataloader), config=val_conifg
#     )
#     tf_scheduler = make_teacher_force_scheduler(steps_per_epoch=len(train_dataloader), config=val_conifg)

#     # model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, metrics = make(config)
#     print(model)

#     train(
#         model,
#         train_dataloader,
#         val_dataloader,
#         loss_fn,
#         optimizer,
#         scheduler,
#         val_conifg,
#         teacher_force_scheduler=tf_scheduler,
#     )

#     torch.save(model.state_dict(), "checkpoints/bert_tiny.pt")

# %% [markdown]
# ## Evaluation

# %%
val_conifg =  {
  "checkpoint_name": "bert_tiny",
  "model_type": "encoder_decoder",
  "initialize_cross_attention": True,
  "batch_size": 256
}

# %%
# train_data = datasets.load_from_disk("data/processed//train/")
val_data = datasets.load_from_disk("data/processed/validation/").select(range(10))
test_data = datasets.load_from_disk("data/processed/test/").select(range(10))

tokenizer = make_tokenizer(val_conifg)
model = make_model(val_conifg, tokenizer)
model.load_state_dict(torch.load("checkpoints/bert_tiny.pt"))

results = evaluate(model, tokenizer, val_data, test_data, val_conifg)

# %% [markdown]
# # [Task 7] Error Analysis


