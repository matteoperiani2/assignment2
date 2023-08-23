from typing import List, Optional, Union
import numpy as np

import torch
from torchmetrics.classification import MulticlassF1Score

import datasets
from accelerate import Accelerator

from .preprocessing import CoQADatasetPreprocessing, idx_to_answer
from .squad_f1 import compute_f1
from .train import DynamicPaddingCollatorForSeq2Seq
from .utils import batched_function, logits_to_class, prepare_model_inputs
from .config import Config

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


def labels_to_answer(labels: torch.Tensor, tokenizer, ignore_index=-100) -> str:
    labels[labels == ignore_index] = tokenizer.pad_token_id
    answer = tokenizer.decode(labels, skip_special_tokens=True)
    return answer


def evaluate_answer(example: dict):
    return {"answer_f1": compute_f1(example["answer"], example["pred_answer"])}


def evaluate_rationale_f1(example: dict):
    return {
        "rationale_f1": per_token_f1_metric(
            logits_to_class(example["rationale_logits"]).long(),
            example["rationale_labels"].long(),
        )
    }


def evaluate_model(model, tokenizer, dataset: datasets.Dataset, config):
    accelerator = Accelerator(mixed_precision=config.mixed_precision)
    model = accelerator.prepare(model)
    model.eval()

    collator = DynamicPaddingCollatorForSeq2Seq(tokenizer, model)
    dataset = dataset.map(
        lambda example: generate_answer_from_input_tensors(
            model, tokenizer, example, collator
        ),
        batched=True,
        batch_size=4,
    )
    dataset = dataset.map(
        lambda example: {
            "answer": labels_to_answer(example["labels"], tokenizer=tokenizer)
        }
    )

    # outputs = outputs.select_columns(["source", "passage", "question", "rationale", "answer", "pred_answer", "answer_type", 'yng_logits', 'rationale_logits'])

    dataset = dataset.map(evaluate_answer)
    dataset = dataset.map(evaluate_rationale_f1)

    yng_data = dataset.select_columns(["yng_logits", "yng_labels"])
    yng_data = yng_data.map(
        lambda example: {"yng_logits": logits_to_class(example["yng_logits"])}
    )
    yng_f1 = macro_f1(yng_data["yng_logits"], yng_data["yng_labels"])

    rationale_f1 = np.mean(dataset["rationale_f1"])
    answer_squad_f1 = np.mean(dataset["answer_f1"])

    return dataset, {
        "yng_f1": yng_f1,
        "rationale_f1": rationale_f1,
        "answer_squad_f1": answer_squad_f1,
    }


def evaluate_answers(model, tokenizer, data):
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    preprocessing = CoQADatasetPreprocessing(tokenizer, **CONFIG.preprocessing.__dict__)

    outputs = data.map(
        lambda example: generate_answer(
            model, tokenizer, preprocessing, example["passage"], example["question"]
        ),
        batched=True,
        batch_size=4,
    )

    outputs = outputs.select_columns(
        [
            "source",
            "passage",
            "question",
            "rationale",
            "answer",
            "pred_answer",
            "answer_type",
        ]
    )

    outputs = outputs.map(evaluate_answer)

    return outputs


def generate_answer(
    model,
    tokenizer,
    preprocessing,
    passage: Union[str, List[str]],
    question: Union[str, List[str]],
    history: Optional[Union[str, List[str]]] = None,
) -> List[str]:
    use_history = history is not None
    preprocess = batched_function(preprocessing.preprocess_texts)
    if isinstance(passage, str):
        passage = [passage]
        question = [question]
        history = [history]

    inputs = {
        "id": list(range(len(passage))),
        "passage": passage,
        "question": question,
    }
    if use_history:
        inputs["history"] = history

    inputs = preprocess(inputs)
    inputs = preprocessing.process_data_to_model_inputs(
        inputs, add_history=use_history, padding="max_length"
    )
    inputs = inputs.convert_to_tensors("pt")

    return generate_answer_from_input_tensors(model, tokenizer, inputs)


def generate_answer_from_input_tensors(model, tokenizer, inputs, collator):
    features = [dict(zip(inputs.keys(), values)) for values in zip(*inputs.values())]
    features = collator(features)

    encoder = model.get_encoder()
    encoder_inputs = prepare_model_inputs(encoder, features)

    inputs = prepare_model_inputs(model, features)
    inputs.pop("decoder_input_ids", None)

    with torch.no_grad():
        encoder_outputs = model.encoder(**encoder_inputs, return_dict=True)
        outputs = model.generate(**inputs)

    answer_types = logits_to_class(encoder_outputs["yng_logits"], task="multiclass")
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for i in range(len(output_str)):
        if answer_types[i] != 2:
            output_str[i] = idx_to_answer(answer_types[i])

    return {
        "pred_answer": output_str,
        "yng_logits": encoder_outputs["yng_logits"],
        "rationale_logits": encoder_outputs["rationale_logits"],
    }
