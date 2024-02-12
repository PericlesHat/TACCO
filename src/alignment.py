# ---- coding: utf-8 ----
# @author: Ziyang Zhang et al.

import torch
import torch.nn as nn

def weighted_cluster_average(node_Q, node_feat, edge_Q, edge_feat):

    node_weight_sum = torch.sum(node_Q, dim=0)
    node_cluster_feat = torch.mm(node_Q.T, node_feat)
    node_cluster_feat /= node_weight_sum.unsqueeze(1)

    edge_weight_sum = torch.sum(edge_Q, dim=0)
    edge_cluster_feat = torch.mm(edge_Q.T, edge_feat)
    edge_cluster_feat /= edge_weight_sum.unsqueeze(1)

    return node_cluster_feat, edge_cluster_feat


class PredictionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PredictionHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, x):
        return self.fc(x)

    def triplet_loss(self, anchor, positive, negative, margin=2):
        cos_sim_pos = self.cosine_similarity(anchor.unsqueeze(0), positive.unsqueeze(0))
        cos_sim_neg = self.cosine_similarity(anchor.unsqueeze(0), negative.unsqueeze(0))

        distance_positive = 1 - cos_sim_pos
        distance_negative = 1 - cos_sim_neg

        losses = torch.relu(distance_positive - distance_negative + margin)
        return losses.mean()

    def build_loss(self, node_cluster_feat, edge_cluster_feat):

        embedded_X_s = self(node_cluster_feat)
        embedded_X_t = self(edge_cluster_feat)

        # O(K^2)
        total_loss = 0
        for i in range(embedded_X_s.shape[0]):
            # get all distance (neg cos sim) to anchor: O(K)
            distances = [-self.cosine_similarity(embedded_X_s[i].unsqueeze(0), t.unsqueeze(0)) for t in embedded_X_t]
            positive_index = distances.index(min(distances))
            # accumulate neg
            for j in range(embedded_X_t.shape[0]):
                if j != positive_index:
                    total_loss += self.triplet_loss(embedded_X_s[i], embedded_X_t[positive_index], embedded_X_t[j])

        total_loss /= embedded_X_s.shape[0]
        return total_loss