"""Group Distributionally Robust Optimization (GroupDRO) utilities (Milestone 7F).

GroupDRO (Sagawa et al., 2020, "Distributionally Robust Neural Networks for Group
Shifts") minimizes the *worst-group* loss instead of the average loss. Instead of the
plain-ERM objective ``mean_i CE(f(x_i), y_i)``, it maintains a probability vector ``q``
over the groups and optimizes ``sum_g q_g * L_g`` where ``L_g`` is the mean cross-entropy
over the examples of group ``g`` in the batch. ``q`` is nudged toward the currently
worst-performing groups by the multiplicative (exponentiated-gradient) update

    q_g <- q_g * exp(step_size * detached_L_g)
    q   <- q / sum(q)

so groups with a higher (detached) loss get a larger weight, making the model focus on
them. Here the **groups are the acquisition centers** {0, 3, 4} — the domain-shift axis
of this project. The center-stratified caches (Milestone 7F-1) make every ``(center,
label)`` cell balanced, so a center-grouped objective is *sound* (center is decorrelated
from label) rather than learning ``center == label`` as the old confounded cache would.

Design notes / edge cases:
  * Base loss is cross-entropy (``reduction="none"`` internally so per-example losses can
    be pooled per group).
  * A group **absent from the current batch** contributes no examples: its per-batch loss
    is undefined, so its ``q`` weight is **not updated** for that batch (it is carried
    forward unchanged, then the whole vector is renormalized). This matches the standard
    GroupDRO handling of missing groups in a minibatch.
  * ``q`` lives on the model device and persists across batches/epochs (it is state of the
    loss object). It is initialized uniform (``1 / n_groups``).
  * The robust loss returned is differentiable in the model parameters through ``L_g``;
    only the ``q`` update uses **detached** group losses (``q`` is not part of the graph).

This module is training-only. It does no inference, no calibration, no ood usage.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_group_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
    group_ids: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean cross-entropy per group for one batch.

    Parameters
    ----------
    logits:
        Model outputs ``[B, C]`` (raw logits).
    labels:
        Integer targets ``[B]`` in ``{0, 1}``.
    groups:
        Integer group id per example ``[B]`` (here: the acquisition ``center``).
    group_ids:
        The full ordered list of group ids to report over (e.g. ``[0, 3, 4]``). Groups
        not present in the batch get a mean loss of ``0.0`` and ``present=False`` /
        ``count=0`` — the caller must NOT update those groups' weights.

    Returns
    -------
    (group_loss, group_present, group_count):
        * ``group_loss``   — ``[G]`` mean CE per group (0.0 for absent groups; the value
          for present groups is differentiable in the model parameters),
        * ``group_present``— ``[G]`` bool mask, True where the group had >= 1 example,
        * ``group_count``  — ``[G]`` long, examples per group in this batch.
        ``G == len(group_ids)`` and rows follow the ``group_ids`` order.
    """
    device = logits.device
    per_example = F.cross_entropy(logits, labels, reduction="none")  # [B], grad-tracked

    g = len(group_ids)
    group_loss = torch.zeros(g, device=device, dtype=per_example.dtype)
    group_present = torch.zeros(g, device=device, dtype=torch.bool)
    group_count = torch.zeros(g, device=device, dtype=torch.long)

    for gi, gid in enumerate(group_ids):
        mask = groups == int(gid)
        count = int(mask.sum().item())
        group_count[gi] = count
        if count > 0:
            group_loss[gi] = per_example[mask].mean()
            group_present[gi] = True
    return group_loss, group_present, group_count


class GroupDROLoss(nn.Module):
    """Group-DRO robust loss with an exponentiated-gradient ``q`` update.

    Maintains a persistent probability vector ``q`` over ``group_ids`` (uniform init).
    Call the module with ``(logits, labels, groups)`` per batch; it computes the per-group
    mean cross-entropy, updates ``q`` multiplicatively from the **detached** group losses
    (skipping groups absent from the batch), renormalizes ``q``, and returns the robust
    loss ``sum_g q_g * L_g`` (differentiable in the model parameters).

    Parameters
    ----------
    group_ids:
        Ordered group ids (here center ids ``[0, 3, 4]``).
    step_size:
        GroupDRO ``q`` update step size (``eta_q``). Default 0.01.
    normalize_group_weights:
        If True (default), renormalize ``q`` to sum to 1 after each update.
    device:
        Device to place ``q`` on (defaults to CPU; ``to()`` moves it with the module).
    """

    def __init__(
        self,
        group_ids: Sequence[int],
        step_size: float = 0.01,
        normalize_group_weights: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.group_ids = [int(g) for g in group_ids]
        self.n_groups = len(self.group_ids)
        self.step_size = float(step_size)
        self.normalize_group_weights = bool(normalize_group_weights)
        # q is buffer state (persists, moves with .to(device), not a learnable parameter).
        q_init = torch.full((self.n_groups,), 1.0 / self.n_groups, dtype=torch.float32)
        self.register_buffer("q", q_init if device is None else q_init.to(device))
        # Bookkeeping of the most recent batch (for logging).
        self.last_group_loss: Optional[torch.Tensor] = None
        self.last_group_present: Optional[torch.Tensor] = None
        self.last_group_count: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Reset ``q`` to uniform (call before training a fresh model)."""
        with torch.no_grad():
            self.q.fill_(1.0 / self.n_groups)
        self.last_group_loss = None
        self.last_group_present = None
        self.last_group_count = None

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        groups: torch.Tensor,
    ) -> torch.Tensor:
        """Return the robust loss for one batch and update ``q`` in place.

        ``q`` is updated only for groups present in the batch (absent groups keep their
        prior weight), then renormalized. The returned scalar is differentiable in the
        model parameters through the present groups' losses.
        """
        group_loss, group_present, group_count = compute_group_losses(
            logits, labels, groups, self.group_ids
        )
        self.last_group_loss = group_loss.detach()
        self.last_group_present = group_present.detach()
        self.last_group_count = group_count.detach()

        # --- q update: exponentiated gradient on DETACHED group losses ---------- #
        with torch.no_grad():
            detached = group_loss.detach()
            present = group_present
            # Only present groups are updated; absent groups keep their weight.
            factors = torch.ones_like(self.q)
            factors[present] = torch.exp(self.step_size * detached[present])
            self.q.mul_(factors)
            if self.normalize_group_weights:
                total = self.q.sum()
                if float(total) > 0:
                    self.q.div_(total)

        # --- robust loss: sum_g q_g * L_g (grad flows through L_g) -------------- #
        robust_loss = torch.sum(self.q.detach() * group_loss)
        return robust_loss

    def q_dict(self) -> dict[int, float]:
        """Current ``q`` as ``{group_id: weight}`` (Python floats)."""
        vals = self.q.detach().cpu().tolist()
        return {gid: float(v) for gid, v in zip(self.group_ids, vals)}

    def last_group_loss_dict(self) -> dict[int, Optional[float]]:
        """Most recent per-group mean loss as ``{group_id: loss or None}``.

        ``None`` for groups that were absent from the last batch.
        """
        if self.last_group_loss is None:
            return {gid: None for gid in self.group_ids}
        losses = self.last_group_loss.cpu().tolist()
        present = self.last_group_present.cpu().tolist()
        return {
            gid: (float(l) if p else None)
            for gid, l, p in zip(self.group_ids, losses, present)
        }


def groups_from_batch(batch, group_field: str, device: torch.device) -> torch.Tensor:
    """Extract an integer group tensor ``[B]`` from a dataloader batch.

    ``BalancedSubsetDataset`` collates metadata ints (e.g. ``center``) either as a tensor
    or a list; this coerces whatever came through into a long tensor on ``device``.
    """
    raw = batch[group_field]
    if torch.is_tensor(raw):
        return raw.to(device=device, dtype=torch.long)
    return torch.tensor([int(v) for v in raw], dtype=torch.long, device=device)


def summarize_group_counts(
    dataset_examples: Iterable[dict], group_ids: Sequence[int]
) -> dict:
    """Count examples per ``(group, label)`` cell over a list of example dicts.

    Returns ``{"per_group": {gid: n}, "per_cell": {(gid,label): n}, "labels": {..}}`` —
    used by the training script's assertions (every center x label cell present).
    """
    per_group = {int(g): 0 for g in group_ids}
    per_cell: dict[tuple[int, int], int] = {}
    labels: dict[int, int] = {}
    for ex in dataset_examples:
        c = int(ex["center"])
        y = int(ex["label"])
        per_group[c] = per_group.get(c, 0) + 1
        per_cell[(c, y)] = per_cell.get((c, y), 0) + 1
        labels[y] = labels.get(y, 0) + 1
    return {"per_group": per_group, "per_cell": per_cell, "labels": labels}
