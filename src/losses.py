
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# 1. FOCAL LOSS
# --------------------------------------------------

class FocalLoss(nn.Module):

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()

        self.gamma = gamma

        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        target_log_probs = log_probs.gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)

        target_probs = probs.gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)

        loss = -(1 - target_probs) ** self.gamma * target_log_probs

        if self.alpha is not None:
            alpha = self.alpha[targets]
            loss = alpha * loss

        return loss.mean()


# --------------------------------------------------
# 2. SUPERVISED CONTRASTIVE LOSS
# --------------------------------------------------


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):

        embeddings = F.normalize(embeddings, dim=1)

        similarity = torch.matmul(
            embeddings, embeddings.T
        ) / self.temperature

        labels = labels.contiguous().view(-1, 1)

        positive_mask = torch.eq(
            labels, labels.T
        ).float()

        batch_size = embeddings.shape[0]

        identity = torch.eye(
            batch_size,
            device=embeddings.device
        )

        positive_mask = positive_mask * (1 - identity)

        # Exclude self-similarity from denominator
        logits = similarity.masked_fill(
            identity.bool(), float("-inf")
        )

        log_probs = similarity - torch.logsumexp(
            logits, dim=1, keepdim=True
        )

        positive_count = positive_mask.sum(dim=1)

        valid = positive_count > 0

        if not valid.any():
            return embeddings.sum() * 0.0

        mean_positive_log_prob = (
            (positive_mask * log_probs.masked_fill(
                identity.bool(), 0.0
            )).sum(dim=1)[valid]
            / positive_count[valid]
        )

        return -mean_positive_log_prob.mean()



# --------------------------------------------------
# 3. CONSISTENCY LOSS
# --------------------------------------------------

class ConsistencyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits_original, logits_augmented):

        # Convert logits into probabilities
        probs_original = F.softmax(
            logits_original, dim=1
        )

        probs_augmented = F.softmax(
            logits_augmented, dim=1
        )

        # Encourage similar predictions
        loss = F.mse_loss(
            probs_original,
            probs_augmented
        )

        return loss
