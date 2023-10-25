import os
import torch

import numpy as np


import transformers
import datasets

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from .preprocessing import CoQADatasetPreprocessing

class LinearScheduler:
    def __init__(
        self, total_iters:int, start_value:float=1., end_value:float=0., fraction=0.7
    ):

        self.start_value = float(start_value)
        self.end_value = float(end_value)
        self.total_iters = total_iters
        self.fraction = fraction
        self._total_iters = fraction * total_iters
        self.current_step = 0

    def step(self):
        self.current_step += 1

    def get_value(self):
        if self.current_step > self._total_iters:
            return self.end_value
         
        return (
            self.start_value
            + (self.end_value - self.start_value)
            / self._total_iters
            * self.current_step
        )

class DummyScheduler:
    def step(self):
        pass

    def get_value(self):
        return 0.

class DummyLRScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer

    def step(self):
        None

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {}


@dataclass
class DynamicPaddingCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: transformers.PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, transformers.utils.PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        # We have to pad the labels and other features not in  `tokenizer.model_input_names` before calling `tokenizer.pad`
        # as `tokenizer.pad` method will pad only features in `tokenizer.model_input_names`
        tokenizer_input_names = set(self.tokenizer.model_input_names)
        for feature_name in features[0].keys():
            if feature_name not in tokenizer_input_names and isinstance(
                features[0][feature_name], list
            ):
                if feature_name.endswith("labels"):
                    self.pad_feature(feature_name, features, self.label_pad_token_id)
                else:
                    self.pad_feature(feature_name, features)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            "labels" in features
            and "decoder_input_ids" not in features
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

        return features

    def pad_feature(self, feature_name, features, pad_id=0):
        items = (
            [feature[feature_name] for feature in features]
            if feature_name in features[0].keys()
            else None
        )
        # We have to pad the feature before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if items is not None:
            max_item_length = max(len(l) for l in items)
            if self.pad_to_multiple_of is not None:
                max_item_length = (
                    (max_item_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [pad_id] * (max_item_length - len(feature[feature_name]))
                if isinstance(feature[feature_name], list):
                    feature[feature_name] = (
                        feature[feature_name] + remainder
                        if padding_side == "right"
                        else remainder + feature[feature_name]
                    )
                elif padding_side == "right":
                    feature[feature_name] = np.concatenate(
                        [feature[feature_name], remainder]
                    )
                else:
                    feature[feature_name] = np.concatenate(
                        [remainder, feature[feature_name]]
                    )


def save_checkpoint(
    model, optimizer, scheduler, epoch, step, checkpoint_counter, checkpoint_path
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "checkpoint_counter": checkpoint_counter,
    }

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint["epoch"]
    step = checkpoint["step"]
    checkpoint_counter = checkpoint["checkpoint_counter"]

    print(f"Loaded checkpoint from '{checkpoint_path}'")

    return model, optimizer, scheduler, epoch, step, checkpoint_counter

def prepare_inputs_for_train(
    dataset: datasets.DatasetDict,
    checkpoints: Dict[str, str],
    filename_fn,
    add_history=False,
    num_processes=None,
    verbose=True,
    **preprocessing_kwargs,
):
    for name, checkpoint in checkpoints.items():
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
        preprocessing = CoQADatasetPreprocessing(tokenizer, **preprocessing_kwargs)

        if verbose:
            print("Preparing inputs for", name, "...")

        if not os.path.exists(filename_fn(name)):
            dataset_ = dataset.map(
                preprocessing.process_data_to_model_inputs,
                fn_kwargs={"add_history": add_history},
                batched=True,
                remove_columns=dataset["train"].column_names,
                num_proc=num_processes,
            )

            dataset_.save_to_disk(filename_fn(name))
            del dataset_

        if verbose:
            dataset_ = datasets.load_from_disk(filename_fn(name))
            print(dataset_)
            print()
            print("Showing some input examples:")
            decoded_inputs = tokenizer.batch_decode(dataset_["train"][:5]["input_ids"])
            for decoded in decoded_inputs:
                print(decoded)
            print()
            del dataset_