from typing import List, Optional, Union

import torch
from torchmetrics.classification import MulticlassF1Score

import datasets
from accelerate import Accelerator

from src.preprocessing import CoQADatasetPreprocessing, idx_to_answer
from src.squad_f1 import compute_f1
from src.train import DynamicPaddingCollatorForSeq2Seq
from src.utils import batched_function, logits_to_class, prepare_model_inputs
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


def labels_to_answer(labels: torch.Tensor, tokenizer, ignore_index=-100) -> str:
    labels[labels == ignore_index] = tokenizer.pad_token_id
    answer = tokenizer.decode(labels, skip_special_tokens=True)
    return answer


def pad_input_tensors(inputs, collator):
    features = [dict(zip(inputs.keys(), values)) for values in zip(*inputs.values())]
    features = collator(features)

    return features


def evaluate_answer(example: dict):
    return {"answer_f1": compute_f1(example["answer"], example["pred_answer"])}


def evaluate_rationale_f1(example: dict):
    return {
        "rationale_f1": per_token_f1_metric(
            logits_to_class(example["rationale_logits"], task="binary").long(),
            example["rationale_labels"].long(),
        )
    }


def evaluate_model(model, tokenizer, dataset: datasets.Dataset, config):
    accelerator = Accelerator(mixed_precision=config.mixed_precision, cpu=config.cpu)
    model = accelerator.prepare(model)
    model.eval()

    collator = DynamicPaddingCollatorForSeq2Seq(tokenizer, model)

    dataset = dataset.map(
        lambda example: pad_input_tensors(example, collator),
        batched=True,
        batch_size=32,
        load_from_cache_file=False,
    )

    dataset = dataset.with_format("torch", device=model.device)

    dataset = dataset.map(
        lambda example: generate_answer(model, tokenizer, example),
        batched=True,
        batch_size=32,
        load_from_cache_file=False,
    )

    dataset = dataset.map(
        lambda example: {
            "answer": labels_to_answer(example["labels"], tokenizer=tokenizer)
        },
        load_from_cache_file=False,
    )

    # outputs = outputs.select_columns(["source", "passage", "question", "rationale", "answer", "pred_answer", "answer_type", 'yng_logits', 'rationale_logits'])

    dataset = dataset.map(evaluate_answer, load_from_cache_file=False)

    dataset = dataset.map(
        evaluate_rationale_f1, batched=True, batch_size=32, load_from_cache_file=False
    )

    yng_data = dataset.select_columns(["yng_logits", "yng_label"])
    yng_data = yng_data.map(
        lambda example: {
            "pred_yng_label": logits_to_class(example["yng_logits"], task="multiclass")
        },
        batched=True,
        batch_size=32,
        load_from_cache_file=False,
    )
    macro_f1_ = macro_f1.to(model.device)
    yng_f1 = macro_f1_(yng_data["pred_yng_label"], yng_data["yng_label"]).item()

    rationale_f1 = torch.mean(dataset["rationale_f1"]).item()
    answer_squad_f1 = torch.mean(dataset["answer_f1"]).item()

    dataset.reset_format()
    return dataset, {
        "yng_f1": yng_f1,
        "rationale_f1": rationale_f1,
        "answer_squad_f1": answer_squad_f1,
    }


def evaluate_model_raw_data(model, tokenizer, data):
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    preprocessing = CoQADatasetPreprocessing(tokenizer, **CONFIG.preprocessing.__dict__)

    outputs = data.map(
        lambda example: generate_answer_from_raw_data(
            model, tokenizer, preprocessing, example["passage"], example["question"]
        ),
        batched=True,
        batch_size=32,
    )

    outputs = outputs.map(evaluate_answer)

    return outputs.select_columns(
        [
            "source",
            "passage",
            "question",
            "rationale",
            "answer",
            "pred_answer",
            "answer_type",
            "answer_squad_f1",
        ]
    )


def generate_answer_from_raw_data(
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

    return generate_answer(model, tokenizer, inputs)


def generate_answer(model, tokenizer, inputs):
    encoder = model.get_encoder()
    encoder_inputs = prepare_model_inputs(encoder, inputs)

    inputs = prepare_model_inputs(model, inputs)
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
