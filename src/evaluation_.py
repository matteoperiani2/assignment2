import inspect
from typing import Dict, List, Literal, Optional, Protocol, Sequence, Union
import numpy as np

import torch
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import MulticlassF1Score

import datasets

from preprocessing import CoQADatasetPreprocessing, idx_to_answer
from squad_f1 import compute_f1, squad_f1
from train import DynamicPaddingCollatorForSeq2Seq
from utils import batched_function, logits_to_class
from config import Config

CONFIG: Config = Config()


class Metric(Protocol):
    def __call__(self, outputs, targets: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        pass


class LogitsToClassMixin:
    def __init__(self, task: Literal["binary", "multiclass"]) -> None:
        if task == "binary":
            self.logits_to_class = lambda logits: logits > 0.0
        elif task == "multiclass":
            self.logits_to_class = lambda logits: torch.argmax(logits, dim=-1)
        else:
            raise ValueError(
                "Invalid task. Supported values are 'binary' and 'multiclass'."
            )


class AverageAccuracy:
    def __init__(self, ignore_index=-100) -> None:
        self.ignore_index = ignore_index

    def __call__(
        self, preds: torch.LongTensor, labels: torch.LongTensor
    ) -> torch.FloatTensor:
        assert preds.dtype == torch.long, "Input `preds` must be a long tensor"
        assert labels.dtype == torch.long, "Input `labels` must be a long tensor"

        valid_labels = labels != self.ignore_index
        match = torch.sum((preds == labels) * valid_labels, dim=-1)
        total = torch.sum(valid_labels, dim=-1)
        per_sample_accuracy = match / total
        return per_sample_accuracy.mean().item()


class AverageAccuracyWithLogits(AverageAccuracy, LogitsToClassMixin):
    def __init__(
        self, task: Literal["binary", "multiclass"], ignore_index=-100
    ) -> None:
        AverageAccuracy.__init__(self, ignore_index=ignore_index)
        LogitsToClassMixin.__init__(self, task=task)

    def __call__(self, logits, labels: torch.Tensor) -> torch.FloatTensor:
        return super().__call__(self.logits_to_class(logits).long(), labels.long())


class AverageMacroF1:
    def __init__(self, num_classes, ignore_index=-100, **kwargs) -> None:
        self.num_classes = num_classes
        self.f1_metric = MulticlassF1Score(
            num_classes=num_classes,
            average="macro",
            multidim_average="samplewise",
            ignore_index=ignore_index,
        )

    def __call__(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.FloatTensor:
        """
        Compute the average over samples of the average over sequence of the macro F1 score.

        Args:
            preds (torch.Tensor): Output preds with shape (batch_size, seq_length).
            labels (torch.Tensor): Output labels with shape (batch_size, seq_length).

        Returns:
            float: Average of the average F1 score.
        """

        f1_score = self.f1_metric(preds.long(), labels.long())
        return f1_score.mean().item()


class AverageMacroF1WithLogits(AverageMacroF1, LogitsToClassMixin):
    def __init__(
        self, task: Literal["binary", "multiclass"], num_classes, ignore_index=-100
    ) -> None:
        AverageMacroF1.__init__(
            self, num_classes=num_classes, ignore_index=ignore_index
        )
        LogitsToClassMixin.__init__(self, task=task)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.FloatTensor:
        """
        Compute the average over samples of the average over sequence of the macro F1 score.

        Args:
            logits (torch.Tensor): Output logits with shape (batch_size, seq_length, num_classes).
            labels (torch.Tensor): Output labels with shape (batch_size, seq_length).

        Returns:
            float: Average of the average F1 score.
        """
        return super().__call__(self.logits_to_class(logits), labels)


class EncoderDecoderRationaleAccuracy(Metric):
    def __init__(self) -> None:
        self.rationale_accuracy = AverageAccuracyWithLogits(task="binary")

    def __call__(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> torch.FloatTensor:
        return self.rationale_accuracy(
            outputs["encoder_rationale_logits"], targets["rationale_labels"]
        )


class EncoderRationaleAccuracy(Metric):
    def __init__(self) -> None:
        self.rationale_accuracy = AverageAccuracyWithLogits(task="binary")

    def __call__(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> torch.FloatTensor:
        return self.rationale_accuracy(
            outputs["rationale_logits"], targets["rationale_labels"]
        )


class EncoderDecoderRationaleF1(Metric):
    def __init__(self) -> None:
        self.rationale_f1 = AverageMacroF1WithLogits(task="binary", num_classes=2)

    def __call__(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> torch.FloatTensor:
        return self.rationale_f1(
            outputs["encoder_rationale_logits"], targets["rationale_labels"]
        )


class EncoderRationaleF1(Metric):
    def __init__(self) -> None:
        self.rationale_f1 = AverageMacroF1WithLogits(task="binary", num_classes=2)

    def __call__(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> torch.FloatTensor:
        return self.rationale_f1(
            outputs["rationale_logits"], targets["rationale_labels"]
        )


class EncoderYNGAccuracy(Metric):
    def __init__(self) -> None:
        self.yng_accuracy = AverageAccuracyWithLogits(task="multiclass")

    def __call__(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> torch.FloatTensor:
        return self.yng_accuracy(outputs["yng_logits"], targets["yng_label"])


class EncoderYNGF1(Metric):
    def __init__(self) -> None:
        self.yng_f1 = AverageMacroF1WithLogits(task="multiclass", num_classes=3)

    def __call__(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> torch.FloatTensor:
        return self.yng_f1(outputs["yng_logits"], targets["yng_label"])


class GenerativeAccuracy(Metric):
    def __init__(self) -> None:
        self.generative_accuracy = AverageAccuracyWithLogits(task="multiclass")

    def __call__(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> torch.FloatTensor:
        return self.generative_accuracy(outputs["logits"], targets["labels"])


class GenerativeSquadF1(Metric):
    def __init__(self, tokenizer, ignore_index=-100) -> None:
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def __call__(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> torch.FloatTensor:
        output_ids = outputs["output_ids"].long()
        predictions = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        labels = targets["labels"].clone().long()
        labels[labels == self.ignore_index] = self.tokenizer.pad_token_id
        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        return squad_f1(predictions=predictions, targets=labels)


class GenerativeSquadF1WithLogits(GenerativeSquadF1, LogitsToClassMixin):
    def __init__(self, tokenizer, ignore_index=-100) -> None:
        LogitsToClassMixin.__init__(self, task="multiclass")
        GenerativeSquadF1.__init__(self, tokenizer=tokenizer, ignore_index=ignore_index)

    def __call__(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> torch.FloatTensor:
        outputs = outputs.clone()
        outputs["output_ids"] = self.logits_to_class(outputs["logits"]).long()
        return super().__call__(outputs, targets)



