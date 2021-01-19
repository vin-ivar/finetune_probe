from typing import Optional, List

from overrides import overrides
import torch
import torch.distributed as dist

from allennlp.common.util import is_distributed
from allennlp.training.metrics.metric import Metric


@Metric.register("per_deprel_scores")
class DeprelScores(Metric):
    def __init__(self, vocab: dict, device, ignore_classes: List[int] = None) -> None:
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._exact_labeled_correct = 0.0
        self._exact_unlabeled_correct = 0.0
        self._total_words = 0.0
        self._total_sentences = 0.0

        self.correct_accumulator = torch.zeros(len(vocab)).long().to(device)
        self.gold_accumulator = torch.zeros(len(vocab)).long().to(device)

        self._ignore_classes: List[int] = ignore_classes or []
        self._vocab = vocab

    def __call__(  # type: ignore
        self,
        predicted_indices: torch.Tensor,
        predicted_labels: torch.Tensor,
        gold_indices: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        metadata = None
    ):
        """
        # Parameters

        predicted_indices : `torch.Tensor`, required.
            A tensor of head index predictions of shape (batch_size, timesteps).
        predicted_labels : `torch.Tensor`, required.
            A tensor of arc label predictions of shape (batch_size, timesteps).
        gold_indices : `torch.Tensor`, required.
            A tensor of the same shape as `predicted_indices`.
        gold_labels : `torch.Tensor`, required.
            A tensor of the same shape as `predicted_labels`.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of the same shape as `predicted_indices`.
        """
        detached = self.detach_tensors(
            predicted_indices, predicted_labels, gold_indices, gold_labels, mask
        )
        predicted_indices, predicted_labels, gold_indices, gold_labels, mask = detached
        device = predicted_indices.device

        if mask is None:
            mask = torch.ones_like(predicted_indices).bool()

        predicted_indices = predicted_indices.long()
        predicted_labels = predicted_labels.long()
        gold_indices = gold_indices.long()
        gold_labels = gold_labels.long()

        # Multiply by a mask denoting locations of
        # gold labels which we should ignore.
        for label in self._ignore_classes:
            label_mask = gold_labels.eq(label)
            mask = mask & ~label_mask

        correct_indices = predicted_indices.eq(gold_indices).long() * mask
        unlabeled_exact_match = (correct_indices + ~mask).prod(dim=-1)
        correct_labels = predicted_labels.eq(gold_labels).long() * mask
        correct_labels_and_indices = correct_indices * correct_labels
        labeled_exact_match = (correct_labels_and_indices + ~mask).prod(dim=-1)
        total_sentences = correct_indices.size(0)
        total_words = correct_indices.numel() - (~mask).sum()

        correct_per_label = correct_indices * gold_labels

        if is_distributed():
            dist.all_reduce(correct_indices, op=dist.ReduceOp.SUM)
            dist.all_reduce(unlabeled_exact_match, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_labels_and_indices, op=dist.ReduceOp.SUM)
            dist.all_reduce(labeled_exact_match, op=dist.ReduceOp.SUM)
            total_sentences = torch.tensor(total_sentences, device=device)
            total_words = torch.tensor(total_words, device=device)
            dist.all_reduce(total_sentences, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_words, op=dist.ReduceOp.SUM)
            total_sentences = total_sentences.item()
            total_words = total_words.item()

        self._unlabeled_correct += correct_indices.sum()
        self._exact_unlabeled_correct += unlabeled_exact_match.sum()
        self._labeled_correct += correct_labels_and_indices.sum()
        self._exact_labeled_correct += labeled_exact_match.sum()
        self._total_sentences += total_sentences
        self._total_words += total_words

        correct_num, correct_count = torch.unique(correct_per_label, return_counts=True)
        gold_num, gold_count = torch.unique(gold_labels, return_counts=True)
        self.correct_accumulator.index_add_(0, correct_num, correct_count)
        self.gold_accumulator.index_add_(0, gold_num, gold_count)

    def get_metric(
        self,
        reset: bool = False,
    ):
        """
        # Returns

        The accumulated metrics as a dictionary.
        """
        unlabeled_attachment_score = 0.0
        labeled_attachment_score = 0.0
        unlabeled_exact_match = 0.0
        labeled_exact_match = 0.0

        if self._total_words > 0.0:
            unlabeled_attachment_score = float(self._unlabeled_correct) / float(self._total_words)
            labeled_attachment_score = float(self._labeled_correct) / float(self._total_words)
        if self._total_sentences > 0:
            unlabeled_exact_match = float(self._exact_unlabeled_correct) / float(
                self._total_sentences
            )
            labeled_exact_match = float(self._exact_labeled_correct) / float(self._total_sentences)
        if reset:
            self.reset()
        metrics = {
            "UAS": unlabeled_attachment_score,
            "LAS": labeled_attachment_score,
            "UEM": unlabeled_exact_match,
            "LEM": labeled_exact_match,
        }

        for i in range(len(self._vocab)):
            score = (self.correct_accumulator[i] / self.gold_accumulator[i]).item()
            metrics[self._vocab[i]] = score
        metrics.pop('punct')

        return metrics

    @overrides
    def reset(self):
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._exact_labeled_correct = 0.0
        self._exact_unlabeled_correct = 0.0
        self._total_words = 0.0
        self._total_sentences = 0.0
