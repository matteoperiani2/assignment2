from dataclasses import dataclass
from typing import Dict, Protocol, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class Loss(Protocol):
    def __call__(self, outputs, targets: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        pass


class ComputeLoss(Protocol):
    def __call__(
        outputs, targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.FloatTensor, Dict[str, float]]:
        pass


@dataclass
class Criterion:
    name: str
    loss_fn: Loss
    weight: float = 1.0


class UncertaintyLoss(nn.Module):
    def __init__(self, *criteria: Criterion) -> None:
        super(UncertaintyLoss, self).__init__()
        self.criteria = criteria
        weights = [criterion.weight for criterion in self.criteria]
        log_square_sigmas = -np.log(weights)
        self.log_square_sigmas = nn.Parameter(
            torch.tensor(log_square_sigmas, requires_grad=True, dtype=torch.float32)
        )

    def forward(
        self, outputs, targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.FloatTensor, Dict[str, float]]:
        losses = {}
        total_loss = 0.0
        for criterion, log_sigma_square in zip(self.criteria, self.log_square_sigmas):
            loss = criterion.loss_fn(outputs, targets)
            # 1/sigma^2 * L + 2 log sigma
            weight = torch.exp(-log_sigma_square)
            total_loss += weight * loss + log_sigma_square
            losses[f"{criterion.name}_weight"] = weight.item()
            losses[criterion.name] = loss.item()

        return total_loss, losses


def encoder_decoder_generative_loss(outputs, targets):
    logits = outputs["logits"]
    labels = targets["labels"]

    return generative_loss(logits, labels)


def generative_loss(logits, labels):
    # swap seq_length with vocabulary dimension
    logits = torch.transpose(logits, 1, 2)
    loss = F.cross_entropy(input=logits, target=labels, reduction="none")
    n_tokens_per_sample = torch.sum(labels != -100, dim=-1)
    loss = torch.sum(loss, dim=-1) / n_tokens_per_sample
    return torch.mean(loss)


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

class RationaleLoss:
    def __init__(
        self,
        max_rationale_length,
        rationale_logits_name="rationale_logits",
        rationale_labels_name="rationale_labels",
        passage_mask_name="passage_mask",
    ) -> None:
        self.max_rationale_length = max_rationale_length
        self.rationale_logits_name = rationale_logits_name
        self.rationale_labels_name = rationale_labels_name
        self.passage_mask_name = passage_mask_name

    def __call__(self, outputs, targets: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        """
        li = w * BCE(y_pred_i, y_true_i)
        , where w = w_positive if y_true_i is positive
                w = w_negative if y_true_i is negative
        w_positive = totals / positives
        w_negative = totals / negatives
        , where totals, positives and negatives are computed for each sequence

        Ls = sum_i=1..seq_length li / sum(w_i)
        L = sum_s=1..N ls / N,
        , where N is the #sequences whose rationale length is <= max_rationale_length
        """

        rationale_logits = outputs[self.rationale_logits_name]
        rationale_labels = targets[self.rationale_labels_name]
        passage_mask = targets[self.passage_mask_name]

        rationale_labels = rationale_labels * passage_mask

        rationale_lengths = torch.sum(rationale_labels, dim=-1, keepdim=True)
        valid_rationales = rationale_lengths <= self.max_rationale_length
        n_sequences = torch.sum(valid_rationales)

        totals = torch.sum(passage_mask, -1, keepdim=True)
        positives = torch.sum(rationale_labels, -1, keepdim=True)
        negatives = totals - positives
        weights = torch.where(
            rationale_labels == 1.0, totals / positives, totals / negatives
        )
        weights = torch.where(weights != torch.inf, weights, 0.0)
        weights = weights * passage_mask
        weights = weights / torch.sum(weights, dim=-1, keepdim=True)
        weights = weights * valid_rationales / n_sequences

        loss = F.binary_cross_entropy_with_logits(
            input=rationale_logits,
            target=rationale_labels,
            weight=weights,
            reduction="sum",
        )
        
        return loss


class EncoderDecoderRationaleLoss(RationaleLoss):
    def __init__(self, max_rationale_length):
        super().__init__(
            max_rationale_length, rationale_logits_name="encoder_rationale_logits"
        )


class EncoderRationaleLoss(ComputeLoss):
    def __init__(self, max_rationale_length):
        self.rationale_loss = RationaleLoss(
            max_rationale_length, rationale_logits_name="rationale_logits"
        )

    def __call__(
        self, outputs, targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.FloatTensor, Dict[str, float]]:
        return self.rationale_loss(outputs, targets) , {}


# def composite_loss(*criteria: Criterion) -> ComputeLoss:
#     def __compute_loss(
#         outputs, targets: Dict[str, torch.Tensor]
#     ) -> Tuple[torch.FloatTensor, Dict[str, float]]:
#         losses = {}
#         total_loss = 0.0
#         for criterion in criteria:
#             loss = criterion.loss_fn(outputs, targets)
#             losses[criterion.name] = loss.item()
#             total_loss += criterion.weight * loss

#         return total_loss, losses

#     return __compute_loss
