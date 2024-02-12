# ---- coding: utf-8 ----
# @author: Ziyang Zhang et al.
# @version: v1, Task-guided Co-clustering Framework
# This work partly uses the code from CACHE.


import json
import os
import argparse
import torch.nn as nn
from tqdm import trange
from models import SetGNN
from preprocessing import *
from dec import DEC
from alignment import PredictionHead, weighted_cluster_average
from datetime import datetime
from convert_datasets_to_pygDataset import dataset_Hypergraph
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import f1_score



@torch.no_grad()
def evaluate(model, data, split_idx, eval_func, epoch, method, dname, args):
    model.eval()

    out_score_g_logits, edge_feat, node_feat, weight_tuple = model(data)
    out_g = torch.sigmoid(out_score_g_logits)

    valid_acc, valid_auroc, valid_aupr, valid_f1_macro = eval_func(data.y[split_idx['valid']],
                                                                   out_g[split_idx['valid']], epoch, method, dname,
                                                                   args, threshold=args.threshold)
    test_acc, test_auroc, test_aupr, test_f1_macro = eval_func(data.y[split_idx['test']],
                                                               out_g[split_idx['test']],
                                                               epoch, method, dname, args,
                                                               threshold=args.threshold)


    return valid_acc, valid_auroc, valid_aupr, valid_f1_macro, \
           test_acc, test_auroc, test_aupr, test_f1_macro

def eval_mimic3(y_true, y_pred, epoch, method, dname, args, threshold=0.5):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    pred = np.array(y_pred > threshold).astype(int)
    correct = (pred == y_true)

    total_acc = []
    total_f1 = []
    for i in range(args.num_labels):
        correct = (pred[:, i] == y_true[:, i])
        accuracy = correct.sum() / correct.size
        total_acc.append(accuracy)
        f1_macro = f1_score(y_true[:, i], pred[:, i], average='macro')
        total_f1.append(f1_macro)

    correct = (pred == y_true)
    accuracy = correct.sum() / correct.size
    f1_macro = f1_score(y_true, pred, average='macro')

    total_auc = []
    for i in range(args.num_labels):
        roc_auc = roc_auc_score(y_true[:, i].reshape(-1), y_pred[:, i].reshape(-1))
        total_auc.append(roc_auc)

    roc_auc = roc_auc_score(y_true.reshape(-1), y_pred.reshape(-1))

    total_aupr = []
    for i in range(args.num_labels):
        aupr = average_precision_score(y_true[:, i].reshape(-1), y_pred[:, i].reshape(-1))
        total_aupr.append(aupr)
    aupr = average_precision_score(y_true.reshape(-1), y_pred.reshape(-1))


    return accuracy, roc_auc, aupr, f1_macro


def eval_cradle(y_true, y_pred, epoch, method, dname, args, threshold=0.5):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    pred = np.array(y_pred > threshold).astype(int)
    correct = (pred == y_true)
    accuracy = correct.sum() / correct.size
    f1_macro = f1_score(y_true.reshape(-1), pred.reshape(-1), average="macro")
    roc_auc = roc_auc_score(y_true.reshape(-1), y_pred.reshape(-1))
    aupr = average_precision_score(y_true.reshape(-1), y_pred.reshape(-1))

    return accuracy, roc_auc, aupr, f1_macro




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prop', type=float, default=0.7)
    parser.add_argument('--valid_prop', type=float, default=0.1)
    parser.add_argument('--dname', default='mimic3')
    parser.add_argument('--method', default='AllSetTransformer')
    parser.add_argument('--text', type=int, default=1)  # 1 for encode text
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--cuda', default='1', type=str)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=1e-3, type=float)
    parser.add_argument('--warmup', default=10, type=int)  # 0 for direct training
    parser.add_argument('--LearnFeat', action='store_true')
    parser.add_argument('--All_num_layers', default=1, type=int)  # hyperparameter L
    parser.add_argument('--MLP_num_layers', default=1, type=int)
    parser.add_argument('--MLP_hidden', default=48, type=int)  # hyperparameter d
    parser.add_argument('--num_cluster', type=int, default=5)  # hyperparameter K
    parser.add_argument('--Classifier_num_layers', default=2, type=int)
    parser.add_argument('--Classifier_hidden', default=64, type=int)
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    parser.add_argument('--normtype', default='all_one')  # ['all_one','deg_half_sym']
    parser.add_argument('--add_self_loop', action='store_false')
    parser.add_argument('--normalization', default='ln')  # NormLayer for MLP. ['bn','ln','None']
    parser.add_argument('--num_features', default=0, type=int)  # placeholder
    parser.add_argument('--num_labels', default=25, type=int)  # 25 for mimic and 1 for cradle
    parser.add_argument('--num_nodes', default=100, type=int)  # 7423 for mimic and 12725 for cradle
    parser.add_argument('--feature_dim', default=128, type=int)  # node embedding dim (*2 if use text)
    parser.add_argument('--PMA', action='store_true')
    parser.add_argument('--heads', default=1, type=int)  # attention heads
    parser.add_argument('--output_ ', default=1, type=int)  # Placeholder
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=10)  # hyperparameter α
    parser.add_argument('--beta', type=float, default=0.1)  # hyperparameter β
    parser.add_argument('--remain_percentage', default=0.3, type=float)

    parser.set_defaults(PMA=True)
    parser.set_defaults(add_self_loop=True)
    parser.set_defaults(LearnFeat=True)

    args = parser.parse_args()

    existing_dataset = ['mimic3', 'cradle']

    synthetic_list = ['mimic3', 'cradle']

    dname = args.dname
    p2raw = '../data/raw_data/'
    dataset = dataset_Hypergraph(name=dname, root='../data/pyg_data/hypergraph_dataset/',
                                 p2raw=p2raw, num_nodes=args.num_nodes, text=args.text)
    data = dataset.data
    args.num_features = dataset.num_features
    if args.dname in ['mimic3', 'cradle']:
        # Shift the y label to start with 0
        data.y = data.y - data.y.min()
    if not hasattr(data, 'n_x'):
        data.n_x = torch.tensor([data.x.shape[0]])
    if not hasattr(data, 'num_hyperedges'):
        # note that we assume the he_id is consecutive.
        data.num_hyperedges = torch.tensor(
            [data.edge_index[0].max() - data.n_x[0] + 1])

    if args.method == 'AllSetTransformer':
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        data = norm_contruction(data, option=args.normtype)

    # hypergraph transformer
    model = SetGNN(args, data)
    # node clustering
    node_cluster = DEC(num_cluster=args.num_cluster, feat_dim=args.MLP_hidden)
    # node clustering
    edge_cluster = DEC(num_cluster=args.num_cluster, feat_dim=args.MLP_hidden)
    # contrastive alignment
    predictor = PredictionHead(input_dim=args.MLP_hidden, hidden_dim=args.MLP_hidden*4, output_dim=args.MLP_hidden)

    # put things to device
    if args.cuda != '-1':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # train-valid-test split
    split_idx = rand_train_test_idx(data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
    train_idx = split_idx['train'].to(device)

    model, data, node_cluster, edge_cluster, predictor = (model.to(device), data.to(device),
                                                          node_cluster.to(device), edge_cluster.to(device),
                                                          predictor.to(device))

    criterion = nn.BCELoss()

    model.train()
    model.reset_parameters()
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    node_cluster_optimizer = torch.optim.Adam(node_cluster.parameters(), lr=args.lr, weight_decay=args.wd)
    edge_cluster_optimizer = torch.optim.Adam(edge_cluster.parameters(), lr=args.lr, weight_decay=args.wd)
    predictor_optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr, weight_decay=args.wd)

    # training logs
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('../logs/', current_time)
    os.makedirs(log_dir, exist_ok=True)

    # start training
    with torch.autograd.set_detect_anomaly(True):
        for epoch in trange(args.epochs):
            if epoch < args.warmup:
                """STEP ONE - WARMUP THE TRANSFORMER"""
                model.train()
                model.zero_grad()

                out_score_logits, _, _, _ = model(data)
                out = torch.sigmoid(out_score_logits)

                warmup_loss = criterion(out[train_idx], data.y[train_idx])
                warmup_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                model_optimizer.step()

            else:
                """STEP TWO - TRAIN THE WHOLE MODEL"""
                model.train()
                model.zero_grad()
                node_cluster.train()
                node_cluster.zero_grad()
                edge_cluster.train()
                edge_cluster.zero_grad()
                predictor.train()
                predictor.zero_grad()

                out_score_logits, out_edge_feat, out_node_feat, _ = model(data)
                out = torch.sigmoid(out_score_logits)

                # clustering loss
                node_cluster_loss = node_cluster.loss(out_node_feat, epoch)
                edge_cluster_loss = edge_cluster.loss(out_edge_feat, epoch)

                # classifier loss
                cls_loss = criterion(out[train_idx], data.y[train_idx])

                # alignment loss
                node_Q = node_cluster.get_Q()
                edge_Q = edge_cluster.get_Q()
                node_onehot = node_cluster.predict()
                edge_onehot = edge_cluster.predict()
                node_label = np.argmax(node_onehot.cpu().detach().numpy(), axis=1)
                edge_label = np.argmax(edge_onehot.cpu().detach().numpy(), axis=1)

                # average cluster features
                node_cluster_feat, edge_cluster_feat = weighted_cluster_average(node_Q, out_node_feat, edge_Q,
                                                                                out_edge_feat)
                align_loss = predictor.build_loss(node_cluster_feat, edge_cluster_feat)

                # final loss
                model_loss = cls_loss + args.alpha * (node_cluster_loss + edge_cluster_loss) + args.beta * align_loss

                model_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                model_optimizer.step()
                node_cluster_optimizer.step()
                edge_cluster_optimizer.step()
                predictor_optimizer.step()


            if dname in ['mimic3']:
                eval_function = eval_mimic3
            elif dname in ['cradle']:
                eval_function = eval_cradle
            valid_acc, valid_auroc, valid_aupr, valid_f1_macro, \
                test_acc, test_auroc, test_aupr, test_f1_macro = \
                evaluate(model, data, split_idx, eval_function, epoch, args.method, args.dname, args)

            # training logs
            fname_valid = ''
            fname_test = ''

            if dname == 'mimic3':
                fname_valid = f'mimic3_valid_{args.method}.txt'
                fname_test = f'mimic3_test_{args.method}.txt'
            elif dname == 'cradle':
                fname_valid = f'cradle_valid_{args.method}.txt'
                fname_test = f'cradle_test_{args.method}.txt'

            fname_valid = os.path.join(log_dir, fname_valid)
            fname_test = os.path.join(log_dir, fname_test)
            fname_hyperparameters = os.path.join(log_dir, 'hyperparameters.txt')

            # save hyperparams
            with open(fname_hyperparameters, 'w', encoding='utf-8') as f:
                args_dict = vars(args)
                f.write(json.dumps(args_dict, indent=4))

            # valid set
            with open(fname_valid, 'a+', encoding='utf-8') as f:
                f.write('Epoch: {}, ACC: {:.5f}, AUROC: {:.5f}, AUPR: {:.5f}, F1_MACRO: {:.5f}\n'
                    .format(epoch + 1, valid_acc, valid_auroc, valid_aupr, valid_f1_macro, ))

            # test set
            with open(fname_test, 'a+', encoding='utf-8') as f:
                f.write('Epoch: {}, ACC: {:.5f}, AUROC: {:.5f}, AUPR: {:.5f}, F1_MACRO: {:.5f}\n'
                    .format(epoch + 1, test_acc, test_auroc, test_aupr, test_f1_macro,))

    print(f'Training finished. Logs are saved in {log_dir}.')