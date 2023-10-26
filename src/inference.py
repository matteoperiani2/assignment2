from typing import List, Optional, Union

import datasets
import torch

from src.config import Config
from src.preprocessing import CoQADatasetPreprocessing
from src.models import idx_to_answer, logits_to_class
from src.train import DynamicPaddingCollatorForSeq2Seq
from src.utils import batched_function, filter_model_inputs, prepare_model_inputs

CONFIG: Config = Config()


def generate_predictions_from_train_ready_dataset(
    data: Union[datasets.Dataset, datasets.DatasetDict],
    tokenizer,
    model,
    batch_size=16,
    ignore_encoder_outputs=False,
    ignore_yng_head=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    data = _pad_tensors(data, batch_size=batch_size, tokenizer=tokenizer, model=model)
    outputs = data.map(
        generate_answer_from_input_tensors,
        fn_kwargs=dict(
            tokenizer=tokenizer,
            model=model,
            ignore_encoder_outputs=ignore_encoder_outputs,
            ignore_yng_head=ignore_yng_head,
        ),
        batched=True,
        batch_size=batch_size,
        load_from_cache_file=False,
    )
    
    outputs.reset_format()
    return outputs


def _pad_tensors(dataset: datasets.Dataset, batch_size, tokenizer, model):
    collator = DynamicPaddingCollatorForSeq2Seq(tokenizer, model)

    def pad_batch(inputs):
        inputs = filter_model_inputs(model, inputs)
        features = [
            dict(zip(inputs.keys(), values)) for values in zip(*inputs.values())
        ]
        features = collator(features)

        return features

    # we sort by n_tokens to optimize memory usage
    dataset = dataset.map(lambda example: {"n_tokens": len(example["input_ids"])})
    dataset = dataset.sort("n_tokens", reverse=True)
    dataset = dataset.map(
        pad_batch,
        batched=True,
        batch_size=batch_size,
        load_from_cache_file=False,
    )

    return dataset.with_format("torch", device=model.device)

# def generate_predictions_from_train_ready_dataset(
#     dataset: Union[datasets.Dataset, datasets.DatasetDict],
#     tokenizer,
#     model,
#     batch_size=16,
#     ignore_encoder_outputs=False,
#     ignore_yng_head=False,
# ):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.eval()
#     model.to(device)

#     collator = DynamicPaddingCollatorForSeq2Seq(tokenizer, model)

#     def pad_batch(inputs):
#         inputs = filter_model_inputs(model, inputs)
#         features = [
#             dict(zip(inputs.keys(), values)) for values in zip(*inputs.values())
#         ]
#         features = collator(features)

#         return features

#     def get_answer_str(labels):
#         labels = torch.as_tensor(labels)
#         labels = torch.where(labels == -100, tokenizer.pad_token_id, labels)
#         return tokenizer.batch_decode(labels, skip_special_tokens=True)

#     def generate_predictions(batch):
#         batch = pad_batch(batch)
#         outputs = generate_answer_from_input_tensors(
#             batch,
#             tokenizer=tokenizer,
#             model=model,
#             ignore_encoder_outputs=ignore_encoder_outputs,
#             ignore_yng_head=ignore_yng_head,
#         )
#         outputs["answer"] = get_answer_str(batch["labels"])
#         return outputs

#     # we sort by n_tokens to optimize memory usage
#     dataset = dataset.map(lambda example: {"n_tokens": len(example["input_ids"])})
#     dataset = dataset.sort("n_tokens", reverse=True)

#     outputs = dataset.map(
#         generate_predictions,
#         batched=True,
#         batch_size=batch_size,
#         load_from_cache_file=False,
#     )
#     outputs.reset_format()

#     return outputs

def generate_answer_from_input_tensors(
    inputs,
    tokenizer,
    model,
    ignore_encoder_outputs=False,
    ignore_yng_head=False,
):
    if not ignore_encoder_outputs:
        encoder = model.get_encoder()
        encoder_inputs = prepare_model_inputs(encoder, inputs)
        with torch.no_grad():
            encoder_outputs = encoder(**encoder_inputs, return_dict=True)

        answer_types = logits_to_class(encoder_outputs["yng_logits"], task="multiclass")

    inputs = prepare_model_inputs(model, inputs)
    inputs.pop("decoder_input_ids", None)

    with torch.no_grad():
        model_outputs = model.generate(**inputs)

    output_str = tokenizer.batch_decode(model_outputs, skip_special_tokens=True)
    outputs = {"pred_answer": output_str}

    if not ignore_encoder_outputs:
        if not ignore_yng_head:
            for i in range(len(output_str)):
                if answer_types[i] != 2:
                    outputs["pred_answer"][i] = idx_to_answer(answer_types[i])

            outputs["pred_yng_logits"]: encoder_outputs["yng_logits"]
            outputs["pred_yng_label"] = logits_to_class(
                encoder_outputs["yng_logits"], task="multiclass"
            )

        outputs["pred_rationale_logits"] = encoder_outputs["rationale_logits"]
        outputs["pred_rationale_labels"] = logits_to_class(
            encoder_outputs["rationale_logits"], task="binary"
        )

    return outputs


def generate_predictions(
    data: Union[datasets.Dataset, datasets.DatasetDict],
    tokenizer,
    model,
    use_history=False,
    batch_size=16,
    ignore_encoder_outputs=False,
    ignore_yng_head=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    question_answerer = QuestionAnswerer(tokenizer, model)
    outputs = data.map(
        lambda example: question_answerer.generate_answer(
            example["passage"],
            example["question"],
            history=example["history"] if use_history else None,
            ignore_encoder_outputs=ignore_encoder_outputs,
            ignore_yng_head=ignore_yng_head,
        ),
        batched=True,
        batch_size=batch_size,
        load_from_cache_file=False,
    )
    return outputs


class QuestionAnswerer:
    def __init__(self, tokenizer, model, preprocessing):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocessing = CoQADatasetPreprocessing(
            tokenizer, **CONFIG.preprocessing.__dict__
        )

    def generate_answer(
        self,
        passage: Union[str, List[str]],
        question: Union[str, List[str]],
        history: Optional[Union[str, List[str]]] = None,
        ignore_encoder_outputs=False,
        ignore_yng_head=False,
    ) -> List[str]:
        use_history = history is not None
        preprocess = batched_function(self.preprocessing.preprocess_texts)

        if isinstance(passage, str):
            passage = [passage]
            question = [question]
            history = [history] if use_history else []

        inputs = {
            "id": list(range(len(passage))),
            "passage": passage,
            "question": question,
        }

        if use_history:
            inputs["history"] = history

        inputs = preprocess(inputs)
        inputs = self.preprocessing.process_data_to_model_inputs(
            inputs, add_history=use_history, padding="max_length"
        )
        inputs = inputs.convert_to_tensors("pt")

        return generate_answer_from_input_tensors(
            inputs,
            tokenizer=self.tokenizer,
            model=self.model,
            ignore_encoder_outputs=ignore_encoder_outputs,
            ignore_yng_head=ignore_yng_head,
        )
    
def create_prediction_dataset(tokenizer, model_name, history=None, seed=""):
    raw_prediction_path = CONFIG.dataset.raw_predictions(model_name=model_name, history=history, seed=seed)

    text_dataset = datasets.load_from_disk(CONFIG.dataset.processed_dir)
    prediction_dataset = datasets.load_from_disk(raw_prediction_path)
    text_dataset.set_format("pandas")
    prediction_dataset.set_format("pandas")

    def merge(text_dataset, prediction_dataset):
        prediction_columns = ["id", "turn", "input_ids", "rationale_start", "rationale_end", "rationale_labels", "passage_mask", "yng_label", "pred_answer", "pred_yng_label", "pred_rationale_labels"]
        prediction_columns = [col for col in prediction_columns if col in prediction_dataset.columns]
        prediction_data = prediction_dataset[prediction_columns].copy()
        prediction_data["input"] = tokenizer.batch_decode(prediction_dataset["input_ids"], skip_special_tokens=False)
        prediction_data.drop("input_ids", axis=1, inplace=True)
        prediction_data = prediction_data.set_index(["id", "turn"])
        predictions = text_dataset.join(prediction_data, on=["id", "turn"])
        return predictions
    
    predictions = {}
    for split, pred_dataset in prediction_dataset.items():
        pred = merge(text_dataset[split][:], pred_dataset[:])
        predictions[split] = datasets.Dataset.from_pandas(pred)

    predictions = datasets.DatasetDict(predictions)
    path = CONFIG.dataset.predictions(model_name=model_name, history=history, seed=seed)
    predictions.save_to_disk(path)
