# ---- coding: utf-8 ----
# @author: Ziyang Zhang et al.
# @description: DEC implementation in PyTorch


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class DEC(nn.Module):
    def __init__(self, num_cluster=4, feat_dim=48):
        super(DEC, self).__init__()

        # Initialize parameters and build layers
        self.feat_dim = feat_dim
        self.num_cluster = num_cluster
        # Mean tensor and cluster assignment matrix, using HE initialization
        self.mean = nn.Parameter(torch.Tensor(num_cluster, self.feat_dim))
        init.kaiming_normal_(self.mean, mode='fan_in', nonlinearity='relu')


    """ calculate Q """
    def build_Q(self, node_feat):
        epsilon = 1.0
        Z = node_feat.unsqueeze(1)  # Shape: [N, 1, dim]

        diff = Z - self.mean  # Broadcasting subtraction
        squared_norm = torch.sum(diff ** 2, dim=2)  # Shape: [N, K]
        Q = torch.pow(squared_norm / epsilon + 1.0, -(epsilon + 1.0) / 2.0)
        return Q / torch.sum(Q, dim=1, keepdim=True)


    """ loss for clustering """
    def loss(self, node_feat, epoch):
        if epoch == 0:
            # print('GRACE using KMEANS to init')
            self.init_mean(node_feat)
        self.Q = self.build_Q(node_feat)
        P = self.get_P()
        loss_c = torch.mean(P * torch.log(P / self.Q))
        return loss_c


    """ init mean from KMEANS """
    def init_mean(self, node_feat):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.num_cluster, n_init=10).fit(node_feat.cpu().detach().numpy())
        cluster_centers_tensor = torch.tensor(kmeans.cluster_centers_).to('cuda')
        self.mean = nn.Parameter(cluster_centers_tensor)


    """ calculate P: each epoch, only called outside """
    def get_P(self):
        f_k = torch.sum(self.Q, dim=0)
        numerator = self.Q**2 / f_k
        denominator_terms = self.Q ** 2 / f_k.unsqueeze(0)
        denominator = torch.sum(denominator_terms, dim=1, keepdim=True)
        return numerator / denominator

    """ predict cluster label """
    def predict(self):
        indices = torch.argmax(self.Q, dim=1)
        one_hot = F.one_hot(indices, num_classes=self.Q.shape[1])
        return one_hot

    """ get Q """
    def get_Q(self):
        return self.Q






