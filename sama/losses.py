from __future__ import annotations

import torch
import torch.nn.functional as F

_SHAPES_LOGGED = False
_KL_CHUNK = 128


def _get_visual_mask(batch: dict, image_token_id: int) -> torch.Tensor:
    mm_types = batch.get("mm_token_type_ids")
    if mm_types is not None:
        # Qwen2.5-VL convention: text=0, image=1, video=2.
        return mm_types.eq(1)
    input_ids = batch["input_ids"]
    return input_ids.eq(int(image_token_id))


def _answer_visual_rows(outputs, batch: dict, image_token_id: int) -> list[torch.Tensor]:
    attentions = getattr(outputs, "attentions", None)
    if not attentions or attentions[-1] is None:
        raise RuntimeError(
            "No last-layer attention weights were returned. Load Qwen2.5-VL with "
            "attn_implementation='eager' and call forward(output_attentions=True)."
        )
    # Gather only the (answer x visual) block. Averaging heads over the full [seq, seq]
    # matrix costs O(seq^2) per view and used to dominate peak memory.
    last = attentions[-1]
    labels = batch["labels"]
    visual_mask = _get_visual_mask(batch, image_token_id)
    rows: list[torch.Tensor] = []

    for b in range(last.size(0)):
        answer_pos = labels[b].ne(-100).nonzero(as_tuple=False).flatten()
        visual_pos = visual_mask[b].nonzero(as_tuple=False).flatten()
        if answer_pos.numel() == 0 or visual_pos.numel() == 0:
            raise ValueError(
                f"Sample {b}: answer positions or visual positions are empty. "
                "Check labels and Qwen2.5-VL image-token masks."
            )
        a = last[b].index_select(1, answer_pos).index_select(2, visual_pos)
        a = a.float().mean(dim=0).clamp_min(1e-8)
        global _SHAPES_LOGGED
        if not _SHAPES_LOGGED:
            _SHAPES_LOGGED = True
            print(
                f"[SAMA][shapes] seq={last.size(-1)} answer={answer_pos.numel()} "
                f"visual={visual_pos.numel()}",
                flush=True,
            )
        # Restrict each answer-token row to the visual-token subspace and renormalize.
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        rows.append(a)
    return rows


def structure_js_loss(clean_outputs, aug_outputs, clean_batch: dict, aug_batch: dict, image_token_id: int) -> torch.Tensor:
    clean_rows = _answer_visual_rows(clean_outputs, clean_batch, image_token_id)
    aug_rows = _answer_visual_rows(aug_outputs, aug_batch, image_token_id)
    if len(clean_rows) != len(aug_rows):
        raise ValueError("Clean/augmented batch sizes differ.")

    per_sample = []
    for i, (p, q) in enumerate(zip(clean_rows, aug_rows)):
        clean_targets = clean_batch["labels"][i][clean_batch["labels"][i].ne(-100)]
        aug_targets = aug_batch["labels"][i][aug_batch["labels"][i].ne(-100)]
        if not torch.equal(clean_targets, aug_targets):
            raise ValueError(
                f"Sample {i}: target answer tokenization differs across views. "
                "Keep the reference answer identical and only perturb the input view."
            )
        if p.shape != q.shape:
            raise ValueError(
                f"Sample {i}: answer/visual attention shapes differ ({tuple(p.shape)} vs {tuple(q.shape)}). "
                "Use layout-preserving image augmentation with unchanged dimensions and token count."
            )
        m = 0.5 * (p + q)
        log_p = p.clamp_min(1e-8).log()
        log_q = q.clamp_min(1e-8).log()
        log_m = m.clamp_min(1e-8).log()
        kl_pm = (p * (log_p - log_m)).sum(dim=-1)
        kl_qm = (q * (log_q - log_m)).sum(dim=-1)
        per_sample.append(0.5 * (kl_pm + kl_qm).mean())
    return torch.stack(per_sample).mean()


def symmetric_kl_from_logits(clean_outputs, noisy_outputs, clean_batch: dict, noisy_batch: dict) -> torch.Tensor:
    clean_logits = clean_outputs.logits
    noisy_logits = noisy_outputs.logits
    clean_labels = clean_batch["labels"]
    noisy_labels = noisy_batch["labels"]
    if clean_logits.size(0) != noisy_logits.size(0):
        raise ValueError("Clean/noisy batch sizes differ.")

    per_sample = []
    for i in range(clean_logits.size(0)):
        # Causal LM logits at t-1 predict labels at t.
        clean_target_seq = clean_labels[i, 1:]
        noisy_target_seq = noisy_labels[i, 1:]
        clean_mask = clean_target_seq.ne(-100)
        noisy_mask = noisy_target_seq.ne(-100)
        clean_targets = clean_target_seq[clean_mask]
        noisy_targets = noisy_target_seq[noisy_mask]
        if not torch.equal(clean_targets, noisy_targets):
            raise ValueError(
                f"Sample {i}: clean/noisy target answer tokens do not align. "
                "Only corrupt user-side text; never modify the reference answer."
            )
        if clean_targets.numel() == 0:
            continue

        clean_pos = clean_mask.nonzero(as_tuple=False).flatten()
        noisy_pos = noisy_mask.nonzero(as_tuple=False).flatten()
        # Chunk over answer positions. Materialising [n_answer, vocab] in float32 for
        # log p, log q, p and q at once is what blew up memory on long answers.
        clean_prefix = clean_logits[i, :-1]
        noisy_prefix = noisy_logits[i, :-1]
        kl_sum = clean_logits.new_zeros((), dtype=torch.float32)
        for start in range(0, clean_pos.numel(), _KL_CHUNK):
            c_idx = clean_pos[start:start + _KL_CHUNK]
            n_idx = noisy_pos[start:start + _KL_CHUNK]
            log_p = F.log_softmax(clean_prefix.index_select(0, c_idx).float(), dim=-1)
            log_q = F.log_softmax(noisy_prefix.index_select(0, n_idx).float(), dim=-1)
            kl_sum = kl_sum + (log_p.exp() * (log_p - log_q)).sum()
            kl_sum = kl_sum + (log_q.exp() * (log_q - log_p)).sum()
        per_sample.append(kl_sum / clean_pos.numel())

    if not per_sample:
        return clean_logits.sum() * 0.0
    return torch.stack(per_sample).mean()


def zero_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def total_sama_loss(
    loss_task: torch.Tensor,
    loss_struct: torch.Tensor,
    loss_safe: torch.Tensor,
    loss_robust: torch.Tensor,
    cfg: dict,
) -> torch.Tensor:
    w = cfg["loss"]
    return (
        float(w.get("lambda_task", 1.0)) * loss_task
        + float(w["lambda_struct"]) * loss_struct
        + float(w["lambda_safe"]) * loss_safe
        + float(w["lambda_robust"]) * loss_robust
    )
