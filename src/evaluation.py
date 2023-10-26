import os
import pickle
import re
import numpy as np
from typing import Union
import numpy as np
import torch

from torchmetrics.classification import MulticlassF1Score

import datasets
import tqdm

from src.generate_annotation import AnswerType
from src.squad_f1 import compute_f1
from src.utils import (
    create_dirs_for_file,
    get_column_names,
    load_pickle,
    save_pickle,
    to_padded_tensor,
)
from src.config import Config

CONFIG: Config = Config()


per_token_f1_metric = MulticlassF1Score(
    num_classes=2,
    average="macro",
    multidim_average="samplewise",
    ignore_index=-100,
)

macro_f1 = MulticlassF1Score(
    num_classes=3,
    average="macro",
    ignore_index=-100,
)

f1 = MulticlassF1Score(
    num_classes=3,
    average="none",
    ignore_index=-100,
)


wh = ["what", "when", "where", "which", "who", "how", "whose", "why"]


def evaluate_predictions(predictions: Union[datasets.Dataset, datasets.DatasetDict]):
    features = set(get_column_names(predictions))

    predictions = predictions.map(compute_squad_f1, load_from_cache_file=False)

    if "pred_rationale_labels" in features and "rationale_labels" in features:
        predictions = predictions.map(
            compute_rationale_f1, batched=True, load_from_cache_file=False
        )

    predictions.reset_format()
    return predictions


def compute_summary_metrics(predictions: datasets.Dataset):
    features = set(predictions.column_names)
    results = {
        "tot_squad_f1": compute_avg_f1(predictions),
        "yes_f1": compute_avg_f1(
            predictions,
            lambda ex: ex["answer_type"] == "yes_no"
            and "yes" in re.findall(r"[\w']+", ex["answer"].lower()),
        ),
        "no_f1": compute_avg_f1(
            predictions,
            lambda ex: ex["answer_type"] == "yes_no"
            and "no" in re.findall(r"[\w']+", ex["answer"].lower()),
        ),
        "wh_question_f1": compute_avg_f1(
            predictions,
            lambda ex: any(
                w in re.findall(r"[\w']+", ex["question"].lower()) for w in wh
            ),
        ),
    }

    f1_by_answer_type = compute_f1_by_answer_type(predictions)
    results = results | f1_by_answer_type

    if "rationale_f1" in features:
        rationale_f1 = np.mean(predictions["rationale_f1"])
        results["rationale_f1"] = (rationale_f1, 1.0)

    if "pred_yng_label" in features and "yng_label" in features:
        true_labels = torch.as_tensor(predictions["yng_label"]).long()
        pred_labels = torch.as_tensor(predictions["pred_yng_label"]).long()
        yng_f1_macro = macro_f1(pred_labels, true_labels).item()
        yng_f1 = f1(pred_labels, true_labels).tolist()

        results["yng_f1_macro"] = (yng_f1_macro, 1.0)
        for name, value in zip(["yes", "no", "gen"], yng_f1):
            results[f"yng_{name}_f1"] = (value, 1.0)

    return results




def compute_f1_by_answer_type(predictions: datasets.Dataset):
    results = {}
    for answer_type in AnswerType.list(return_unknown=False):
        result = compute_avg_f1(
            predictions, lambda ex: ex["answer_type"] == answer_type
        )
        results[answer_type + "_f1"] = result

    return results


def compute_avg_f1(predictions: datasets.Dataset, filter_fn=None):
    examples = predictions.filter(
        filter_fn,
        load_from_cache_file=False,
    )
    avg_f1 = np.mean(examples["answer_f1"])
    ratio = len(examples) / len(predictions)
    return avg_f1, ratio


# def evaluate_generation(model, tokenizer, data, config, use_history=False):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.eval()
#     model.to(device)

#     preprocessing = CoQADatasetPreprocessing(tokenizer, **CONFIG.preprocessing.__dict__)
#     question_answerer = QuestionAnswerer(tokenizer, model, preprocessing)

#     if not use_history:
#         outputs = data.map(
#             lambda example: question_answerer.generate_answer(
#                 example["passage"], example["question"]
#             ),
#             batched=True,
#             batch_size=config.get("generate_batch_size", 0),
#             load_from_cache_file=False,
#         )
#     else:
#         outputs = data.map(
#             lambda example: question_answerer.generate_answer(
#                 example["passage"], example["question"], example["history"]
#             ),
#             batched=True,
#             batch_size=config.get("generate_batch_size", 0),
#             load_from_cache_file=False,
#         )

#     outputs = outputs.map(compute_squad_f1, load_from_cache_file=False)

#     outputs = outputs.select_columns(
#         [
#             "source",
#             "passage",
#             "question",
#             "rationale",
#             "answer",
#             "pred_answer",
#             "answer_type",
#             "answer_f1",
#         ]
#     )

#     yes_answer = outputs.filter(
#         lambda ex: "yes" in re.findall(r"[\w']+", ex["answer"].lower()),
#         load_from_cache_file=False,
#     )
#     no_answer = outputs.filter(
#         lambda ex: "no" in re.findall(r"[\w']+", ex["answer"].lower()),
#         load_from_cache_file=False,
#     )
#     mc_question = outputs.filter(
#         lambda ex: "or" in re.findall(r"[\w']+", ex["question"].lower()),
#         load_from_cache_file=False,
#     )
#     wh_question = outputs.filter(
#         lambda ex: any(w in re.findall(r"[\w']+", ex["question"].lower()) for w in wh),
#         load_from_cache_file=False,
#     )

#     len_data = len(outputs)
#     return outputs, {
#         "tot_squad_f1": (np.mean(outputs["answer_f1"]), 1),
#         "yes_ans_f1": (np.mean(yes_answer["answer_f1"]), len(yes_answer) / len_data),
#         "no_ans_f1": (np.mean(no_answer["answer_f1"]), len(no_answer) / len_data),
#         "mc_quest_f1": (np.mean(mc_question["answer_f1"]), len(mc_question) / len_data),
#         "wh_quest_f1": (np.mean(wh_question["answer_f1"]), len(wh_question) / len_data),
#     }


# def evaluate_conversation(model, tokenizer, df):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.eval()
#     model.to(device)

#     preprocessing = CoQADatasetPreprocessing(tokenizer, **CONFIG.preprocessing.__dict__)
#     question_answerer = QuestionAnswerer(tokenizer, model, preprocessing)

#     conversations_results = []
#     for _, row in tqdm(df.iterrows(), total=df.shape[0], leave=False):
#         passage = row["story"]
#         questions = [q["input_text"] for q in row["questions"]]
#         answers = [a["input_text"] for a in row["answers"]]

#         answer_f1_scores = []  # f1-score of single answers within the dialogue

#         pred_answer = []
#         for quest, answ in zip(questions, answers):
#             outputs = question_answerer.generate_answer(passage, quest)
#             pred_answer.append(outputs["pred_answer"][0])
#             answer_f1_scores.append(
#                 compute_squad_f1({"answer": answ, "pred_answer": pred_answer[-1]})[
#                     "answer_f1"
#                 ]
#             )

#         results = {
#             "source": row["source"],
#             "passage": passage,
#             "questions": questions,
#             "answers": answers,
#             "predicted_answers": pred_answer,
#             "answers_f1_scores": answer_f1_scores,
#             "conversation_f1_score": np.mean(answer_f1_scores),
#         }

#         conversations_results.append(results)

#     return conversations_results


def print_worst_answers(conv_res):
    answers = [
        (ans, ans_f1)
        for idx, answers in enumerate(conv_res["predicted_answers"])
        for ans, ans_f1 in zip(answers, conv_res["answers_f1_scores"][idx])
        if ans_f1 <= conv_res["conversation_f1_score"].min()
    ]
    ans_idx = [idx for idx, obj in enumerate(answers) if obj[1] == min(answers)[1]]

    return np.random.choice(ans_idx, size=5)  # return random 5 worst answers


def compute_squad_f1(example: dict):
    return {"answer_f1": compute_f1(example["answer"], example["pred_answer"])}


def compute_rationale_f1(batch: dict):
    true_labels = to_padded_tensor(batch["rationale_labels"], pad_value=-100).long()
    pred_labels = to_padded_tensor(batch["pred_rationale_labels"]).long()    
    rationale_f1 = per_token_f1_metric(
        pred_labels,
        true_labels,
    )
    # Ensure it is an array, not a scalar
    if rationale_f1.dim() == 0:
        rationale_f1.unsqueeze_(dim=0)
    return {"rationale_f1": rationale_f1.tolist()}




def save_results(results_per_model: dict):
    save_pickle(results_per_model, CONFIG.dataset.results)


def load_results():
    return load_pickle(CONFIG.dataset.results)