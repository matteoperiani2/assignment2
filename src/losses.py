from dataclasses import dataclass
from typing import Dict, Protocol, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def apply_reduction(input: torch.Tensor, reduction: str, dim=0):
    if reduction == "none":
        return input
    if reduction == "mean":
        return torch.mean(input, dim=dim)
    if reduction == "sum":
        return torch.sum(input, dim=dim)

    raise ValueError(
        "Invalid reduction. Supported values are 'none', 'mean' and 'sum'."
    )


class Loss(Protocol):
    def __call__(self, outputs, targets: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        pass


class ComputeLoss(Protocol):
    def __call__(
        outputs, targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.FloatTensor, Dict[str, float]]:
        pass


def wrap_loss_fn(
    name: str, loss_fn: Loss
) -> Tuple[torch.FloatTensor, Dict[str, float]]:
    def loss(outputs, targets):
        loss_value = loss_fn(outputs, targets)
        return loss_value, {name: loss_value.item()}

    return loss


@dataclass
class Criterion:
    name: str
    loss_fn: Loss
    weight: float = 1.0


class UncertaintyLoss(nn.Module, ComputeLoss):
    def __init__(self, name: str, loss_fn: Loss, initial_weight: float = 1.0) -> None:
        super(UncertaintyLoss, self).__init__()
        self.name = name
        self.loss_fn = loss_fn
        log_sigma_square = -np.log(initial_weight)
        self.log_sigma_square = nn.Parameter(
            torch.tensor(log_sigma_square, requires_grad=True, dtype=torch.float32)
        )

    def forward(
        self, outputs, targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.FloatTensor, Dict[str, float]]:
        inner_loss = self.loss_fn(outputs, targets)
        # 1/sigma^2 * L + 2 log sigma
        weight = torch.exp(-self.log_sigma_square)
        loss = weight * inner_loss + self.log_sigma_square

        return loss, {
            self.name: inner_loss.item(),
            f"{self.name}_weight": weight.item(),
        }


# class UncertaintyLoss(nn.Module):
#     def __init__(self, *criteria: Criterion) -> None:
#         super(UncertaintyLoss, self).__init__()
#         self.criteria = criteria
#         weights = [criterion.weight for criterion in self.criteria]
#         log_square_sigmas = -np.log(weights)
#         self.log_square_sigmas = nn.Parameter(
#             torch.tensor(log_square_sigmas, requires_grad=True, dtype=torch.float32)
#         )

#     def forward(
#         self, outputs, targets: Dict[str, torch.Tensor]
#     ) -> Tuple[torch.FloatTensor, Dict[str, float]]:
#         losses = {}
#         total_loss = 0.0
#         for criterion, log_sigma_square in zip(self.criteria, self.log_square_sigmas):
#             loss = criterion.loss_fn(outputs, targets)
#             # 1/sigma^2 * L + 2 log sigma
#             weight = torch.exp(-log_sigma_square)
#             total_loss += weight * loss + log_sigma_square
#             losses[f"{criterion.name}_weight"] = weight.item()
#             losses[criterion.name] = loss.item()

#         return total_loss, losses


def generative_loss(
    logits: torch.FloatTensor, labels: torch.IntTensor, reduction: str = "mean"
) -> torch.FloatTensor:
    # swap seq_length with vocabulary dimension
    logits = torch.transpose(logits, 1, 2)  # batch_size x seq_length x vocab
    loss = F.cross_entropy(
        input=logits, target=labels, reduction="none"
    )  # batch_size x seq_length
    n_tokens_per_sample = torch.sum(labels != -100, dim=-1)  # batch_size
    loss = torch.sum(loss, dim=-1) / n_tokens_per_sample  # batch_size
    return apply_reduction(loss, reduction=reduction)


class EncoderDecoderGenerativeLoss(Loss):
    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = reduction

    def __call__(self, outputs, targets: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits = outputs["logits"]
        labels = targets["labels"]

        return generative_loss(logits, labels, reduction=self.reduction)


# class RationaleLoss:
#     def __init__(
#         self,
#         max_rationale_length,
#         rationale_logits_name="rationale_logits",
#         rationale_labels_name="rationale_labels",
#         passage_mask_name="passage_mask",
#     ) -> None:
#         self.max_rationale_length = max_rationale_length
#         self.rationale_logits_name = rationale_logits_name
#         self.rationale_labels_name = rationale_labels_name
#         self.passage_mask_name = passage_mask_name

#     def __call__(self, outputs, targets: Dict[str, torch.Tensor]) -> torch.FloatTensor:
#         rationale_logits = outputs[self.rationale_logits_name]
#         rationale_labels = targets[self.rationale_labels_name]
#         passage_mask = targets[self.passage_mask_name]

#         rationale_labels = rationale_labels * passage_mask
#         totals = torch.sum(passage_mask, -1, keepdim=True)
#         positives = torch.sum(rationale_labels, -1, keepdim=True)
#         negatives = totals - positives
#         weight = torch.where(
#             rationale_labels == 1.0, totals / positives, totals / negatives
#         )
#         weight = torch.where(weight != torch.inf, weight, 0.0)
#         # weight = torch.ones_like(rationale_labels)

#         per_token_loss = F.binary_cross_entropy_with_logits(
#             input=rationale_logits,
#             target=rationale_labels,
#             weight=weight,
#             reduction="none",
#         )

#         per_token_loss = per_token_loss * passage_mask
#         per_sequence_loss = torch.sum(per_token_loss, dim=-1) / torch.sum(
#             weight * passage_mask, dim=-1
#         )

#         rationale_lengths = torch.sum(rationale_labels, -1)
#         valid_rationales = rationale_lengths <= self.max_rationale_length
#         n_sequences = torch.sum(valid_rationales, dim=-1)

#         return torch.sum(per_sequence_loss * valid_rationales, dim=-1) / n_sequences


# class rationale_loss(Loss):
#     def __init__(
#         self,
#         max_rationale_length,
#         rationale_logits_name="rationale_logits",
#         rationale_labels_name="rationale_labels",
#         passage_mask_name="passage_mask",
#     ) -> None:
#         self.max_rationale_length = max_rationale_length
#         self.rationale_logits_name = rationale_logits_name
#         self.rationale_labels_name = rationale_labels_name
#         self.passage_mask_name = passage_mask_name


# def rationale_loss(self, outputs, targets: Dict[str, torch.Tensor]) -> torch.FloatTensor:
def rationale_loss(
    logits: torch.FloatTensor,
    labels: torch.IntTensor,
    passage_mask: torch.IntTensor,
    max_rationale_length: int,
    reduction="mean",
) -> torch.FloatTensor:
    """
    li = w * BCE(y_pred_i, y_true_i)
    , where w = w_positive if y_true_i is positive
            w = w_negative if y_true_i is negative
    w_positive = totals / positives
    w_negative = totals / negatives
    , where totals, positives and negatives are computed for each sequence

    Ls = sum_i=1..seq_length li / sum(w_i)
    L = sum_s=1..N Ls / N,
    , where N is the #sequences whose rationale length is <= max_rationale_length
    """

    # rationale_logits = outputs[self.rationale_logits_name]
    # rationale_labels = targets[self.rationale_labels_name]
    # passage_mask = targets[self.passage_mask_name]

    labels = labels * passage_mask

    rationale_lengths = torch.sum(labels, dim=-1)  # batch_size
    valid_rationales = rationale_lengths <= max_rationale_length

    labels = labels[valid_rationales]
    passage_mask = passage_mask[valid_rationales]
    logits = logits[valid_rationales]

    # n_sequences = torch.sum(valid_rationales)

    totals = torch.sum(passage_mask, -1, keepdim=True)  # N x 1
    positives = torch.sum(labels, -1, keepdim=True)  # N x 1
    negatives = totals - positives  # N x 1
    weights = torch.where(
        labels == 1.0, totals / positives, totals / negatives
    )  # N x seq_length
    weights = torch.where(weights != torch.inf, weights, 0.0)  # N x seq_length
    weights = weights * passage_mask  # N x seq_length
    weights = weights / torch.sum(weights, dim=-1, keepdim=True)  # N x seq_length
    # weights = weights * valid_rationales / n_sequences

    # N x seq_length
    per_token_loss = F.binary_cross_entropy_with_logits(
        input=logits,
        target=labels,
        weight=weights,
        reduction="none",
    )

    loss = torch.sum(per_token_loss, dim=-1)  # N
    return apply_reduction(loss, reduction=reduction)


class EncoderDecoderRationaleLoss(Loss):
    def __init__(self, max_rationale_length: int, reduction: str = "mean") -> None:
        self.max_rationale_length = max_rationale_length
        self.reduction = reduction

    def __call__(self, outputs, targets: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits = outputs["encoder_rationale_logits"]
        labels = targets["rationale_labels"]
        passage_mask = targets["passage_mask"]

        return rationale_loss(
            logits,
            labels,
            passage_mask,
            self.max_rationale_length,
            reduction=self.reduction,
        )


class EncoderRationaleLoss(Loss):
    def __init__(self, max_rationale_length: int, reduction: str = "mean") -> None:
        self.max_rationale_length = max_rationale_length
        self.reduction = reduction

    def __call__(self, outputs, targets: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits = outputs["rationale_logits"]
        labels = targets["rationale_labels"]
        passage_mask = targets["passage_mask"]

        return rationale_loss(
            logits,
            labels,
            passage_mask,
            self.max_rationale_length,
            reduction=self.reduction,
        )


def yes_no_gen_loss(logits, labels, reduction="mean"):
    loss = F.cross_entropy(logits, labels, reduction=reduction)
    return loss


class EncoderDecoderYNGLoss(Loss):
    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = reduction

    def __call__(self, outputs, targets: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits = outputs["encoder_yng_logits"]
        labels = targets["yng_label"]

        return yes_no_gen_loss(
            logits,
            labels,
            reduction=self.reduction,
        )


class EncoderYNGLoss(Loss):
    def __init__(self, max_rationale_length: int, reduction: str = "mean") -> None:
        self.max_rationale_length = max_rationale_length
        self.reduction = reduction

    def __call__(self, outputs, targets: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits = outputs["yng_logits"]
        labels = targets["yng_label"]

        return yes_no_gen_loss(
            logits,
            labels,
            reduction=self.reduction,
        )


class EncoderDecoderLoss(nn.Module):
    def __init__(
        self,
        max_rationale_length,
        yng_loss_weight=1.0,
        rationale_loss_weight=1.0,
        generative_loss_weight=1.0,
    ) -> None:
        super().__init__()

        # self.rationale_loss_fn = UncertaintyLoss(
        #     name="rationale_loss",
        #     loss_fn=EncoderDecoderRationaleLoss(
        #         max_rationale_length=max_rationale_length
        #     ),
        #     initial_weight=rationale_loss_weight,
        # )

        # self.yes_no_gen_loss_fn = wrap_loss_fn(
        #     name="yng_loss", loss_fn=EncoderDecoderYNGLoss()
        # )

        # self.rationale_loss_fn = wrap_loss_fn(
        #     name="rationale_loss",
        #     loss_fn=EncoderDecoderRationaleLoss(
        #         max_rationale_length=max_rationale_length,
        #         reduction="none"
        #     ),
        # )
        # self.generative_loss_fn = wrap_loss_fn(
        #     name="generative_loss",
        #     loss_fn=EncoderDecoderGenerativeLoss(reduction="none"),
        # )

        self.yes_no_gen_loss_fn = EncoderDecoderYNGLoss(reduction="mean")

        self.rationale_loss_fn = EncoderDecoderRationaleLoss(
            max_rationale_length=max_rationale_length, reduction="none"
        )
        self.generative_loss_fn = EncoderDecoderGenerativeLoss(reduction="none")

    def forward(
        self, outputs, targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.FloatTensor, Dict[str, float]]:
        yng_loss = self.yes_no_gen_loss_fn(outputs, targets)

        is_generative = ~targets["yes_no"].bool()
        rationale_loss = self.rationale_loss_fn(outputs, targets)
        rationale_loss = rationale_loss[is_generative]
        rationale_loss = torch.mean(rationale_loss)

        generative_loss = self.generative_loss_fn(outputs, targets)
        generative_loss = generative_loss[is_generative]
        generative_loss = torch.mean(generative_loss)

        total_loss = yng_loss + rationale_loss + generative_loss
        loss_logs = {
            "yng_loss": yng_loss.item(),
            "rationale_loss": rationale_loss.item(),
            "generative_loss": generative_loss.item(),
        }

        # total_loss = yng_loss + (1 - targets["yes_no"]) * (
        #     rationale_loss + generative_loss
        # )
        # loss_logs = yng_loss_logs | rationale_loss_logs | generative_loss_logs

        return total_loss, loss_logs
