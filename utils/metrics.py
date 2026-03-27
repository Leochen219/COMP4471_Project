# utils/metrics.py

import torch


@torch.no_grad()
def compute_retrieval_metrics(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    ks: tuple = (1, 5, 10),
) -> dict:
    """
    图文检索 Recall@K
    image_embeds: [N, D] L2-normalized
    text_embeds:  [N, D] L2-normalized
    假设: image_i ↔ text_i (对角线为正样本)
    """
    # 相似度矩阵 [N, N]
    sim = image_embeds @ text_embeds.t()
    N = sim.size(0)

    metrics: dict = {}

    # ---- Image → Text ----
    diag_i2t = sim.diag().unsqueeze(1)          # [N, 1]
    i2t_ranks = (sim >= diag_i2t).sum(dim=1)    # [N]  (1-indexed)

    for k in ks:
        metrics[f"i2t_R@{k}"] = 100.0 * (i2t_ranks <= k).float().mean().item()

    # ---- Text → Image ----
    diag_t2i = sim.diag().unsqueeze(0)          # [1, N]
    t2i_ranks = (sim >= diag_t2i).sum(dim=0)    # [N]

    for k in ks:
        metrics[f"t2i_R@{k}"] = 100.0 * (t2i_ranks <= k).float().mean().item()

    # In the current evaluation protocol each query has exactly one matched target,
    # so Recall@1 is equivalent to top-1 retrieval accuracy.
    metrics["i2t_acc@1"] = metrics["i2t_R@1"]
    metrics["t2i_acc@1"] = metrics["t2i_R@1"]
    metrics["mean_acc@1"] = (metrics["i2t_acc@1"] + metrics["t2i_acc@1"]) / 2

    # ---- Mean Recall ----
    recall_values = [v for key, v in metrics.items() if "R@" in key]
    metrics["mean_recall"] = sum(recall_values) / len(recall_values)

    return metrics
