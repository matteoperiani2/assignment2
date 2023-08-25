from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

_EPSILON = 1e-7


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
    logits: torch.FloatTensor,
    labels: torch.IntTensor,
    reduction: str = "mean",
    mask: torch.Tensor = None,
) -> torch.FloatTensor:
    if mask is not None:
        logits = logits[mask.bool()]
        labels = labels[mask.bool()]

    # swap seq_length with vocabulary dimension
    logits = torch.transpose(logits, 1, 2)  # batch_size x seq_length x vocab
    loss = F.cross_entropy(
        input=logits, target=labels, reduction="none"
    )  # batch_size x seq_length
    n_tokens_per_sample = torch.sum(labels != -100, dim=-1)  # batch_size
    n_tokens_per_sample = torch.clamp(n_tokens_per_sample, min=_EPSILON)
    loss = torch.sum(loss, dim=-1) / n_tokens_per_sample  # batch_size
    return apply_reduction(loss, reduction=reduction)


class EncoderDecoderGenerativeLoss(Loss):
    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = reduction

    def __call__(
        self,
        outputs,
        targets: Dict[str, torch.Tensor],
        mask: torch.Tensor = None,
    ) -> torch.FloatTensor:
        logits = outputs["logits"]
        labels = targets["labels"]

        return generative_loss(logits, labels, reduction=self.reduction, mask=mask)


# def rationale_loss(self, outputs, targets: Dict[str, torch.Tensor]) -> torch.FloatTensor:
def rationale_loss(
    logits: torch.FloatTensor,
    labels: torch.IntTensor,
    passage_mask: torch.IntTensor,
    max_rationale_length: int,
    reduction="mean",
    mask: torch.Tensor = None,
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
    if mask is not None:
        valid_rationales = valid_rationales & mask.bool()

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
    normalize_factor = torch.clamp(
        torch.sum(weights, dim=-1, keepdim=True), min=_EPSILON
    )
    weights = weights / normalize_factor  # N x seq_length
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

    def __call__(
        self,
        outputs,
        targets: Dict[str, torch.Tensor],
        mask: torch.Tensor = None,
    ) -> torch.FloatTensor:
        logits = outputs["encoder_rationale_logits"]
        labels = targets["rationale_labels"]
        passage_mask = targets["passage_mask"]

        return rationale_loss(
            logits,
            labels,
            passage_mask,
            self.max_rationale_length,
            reduction=self.reduction,
            mask=mask,
        )


class EncoderRationaleLoss(Loss):
    def __init__(self, max_rationale_length: int, reduction: str = "mean") -> None:
        self.max_rationale_length = max_rationale_length
        self.reduction = reduction

    def __call__(
        self,
        outputs,
        targets: Dict[str, torch.Tensor],
        mask: torch.Tensor = None,
    ) -> torch.FloatTensor:
        logits = outputs["rationale_logits"]
        labels = targets["rationale_labels"]
        passage_mask = targets["passage_mask"]

        return rationale_loss(
            logits,
            labels,
            passage_mask,
            self.max_rationale_length,
            reduction=self.reduction,
            mask=mask,
        )


def yes_no_gen_loss(
    logits: torch.FloatTensor,
    labels: torch.IntTensor,
    weight: Optional[torch.FloatTensor] = None,
    reduction="mean",
    mask: torch.Tensor = None,
) -> torch.FloatTensor:
    if mask is not None:
        logits = logits[mask.bool()]
        labels = labels[mask.bool()]

    if weight is not None:
        weight.to(logits.device)

    loss = F.cross_entropy(logits, labels, reduction=reduction)
    return loss


class EncoderDecoderYNGLoss(Loss):
    def __init__(
        self, weight: Optional[torch.FloatTensor] = None, reduction: str = "mean"
    ) -> None:
        self.weight = weight
        self.reduction = reduction

    def __call__(
        self,
        outputs,
        targets: Dict[str, torch.Tensor],
        mask: torch.Tensor = None,
    ) -> torch.FloatTensor:
        logits = outputs["encoder_yng_logits"]
        labels = targets["yng_label"]

        return yes_no_gen_loss(
            logits, labels, weight=self.weight, reduction=self.reduction, mask=mask
        )


class EncoderYNGLoss(Loss):
    def __init__(self, weight: Optional[torch.FloatTensor] = None, reduction: str = "mean") -> None:
        self.weight = weight
        self.reduction = reduction

    def __call__(
        self,
        outputs,
        targets: Dict[str, torch.Tensor],
        mask: torch.Tensor = None,) -> torch.FloatTensor:
        logits = outputs["yng_logits"]
        labels = targets["yng_label"]

        return yes_no_gen_loss(
            logits,
            labels,
            weight=self.weight,
            reduction=self.reduction,
            mask=mask
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

        self.yng_loss_weight = yng_loss_weight
        self.rationale_loss_weight = rationale_loss_weight
        self.generative_loss_weight = generative_loss_weight

        weight = torch.Tensor([1 / 11.0, 1 / 9.0, 1 / 80.0])
        weight = weight / torch.sum(weight)
        weight = None
        self.yes_no_gen_loss_fn = EncoderDecoderYNGLoss(weight=weight)
        self.yes_no_gen_loss_fn = EncoderDecoderYNGLoss()
        self.rationale_loss_fn = EncoderDecoderRationaleLoss(
            max_rationale_length=max_rationale_length
        )
        self.generative_loss_fn = EncoderDecoderGenerativeLoss()

    def forward(
        self, outputs, targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.FloatTensor, Dict[str, float]]:
        yng_loss = self.yes_no_gen_loss_fn(outputs, targets)

        is_generative = ~targets["yes_no"].bool()
        rationale_loss = self.rationale_loss_fn(outputs, targets, mask=is_generative)
        generative_loss = self.generative_loss_fn(outputs, targets, mask=is_generative)

        total_loss = (
            self.yng_loss_weight * yng_loss
            + self.rationale_loss_weight * rationale_loss
            + self.generative_loss_weight * generative_loss
        )
        loss_logs = {
            "yng_loss": yng_loss.item(),
            "rationale_loss": rationale_loss.item(),
            "generative_loss": generative_loss.item(),
        }

        return total_loss, loss_logs
