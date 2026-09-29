"""Classification metrics for the graph classifiers (teacher GIN and LogiX-GIN models).

Accuracy alone is misleading on AIDS (80/20) and BBBP (76/24), so the final
evaluation of every run also reports balanced accuracy, F1 of the minority class and
ROC-AUC. The models output one sigmoid per class; for two classes the ranking score
is ``out[:, 1] - out[:, 0]``.
"""
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


@torch.no_grad()
def predict(model, loader, device):
    """Class outputs and labels over a loader, with the same forward as test_epoch."""
    model.eval()
    outs, ys = [], []
    for data in loader:
        if data.x is None:
            data.x = torch.ones((data.num_nodes, model.num_features))
        if data.y.numel() == 0:
            continue
        out = model(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device), tau=1000)
        outs.append(out.detach().cpu())
        ys.append(data.y.reshape(-1).cpu())
    return torch.cat(outs).numpy(), torch.cat(ys).long().numpy()


def evaluate(model, loader, device, prefix=''):
    """acc, balanced_acc, f1 (minority class), auc (binary only), n."""
    out, y = predict(model, loader, device)
    pred = out.argmax(1)
    res = {'acc': float((pred == y).mean()),
           'balanced_acc': float(balanced_accuracy_score(y, pred)),
           'n': int(len(y))}
    classes = np.unique(y)
    if out.shape[1] == 2 and len(classes) == 2:
        minority = int(np.argmin(np.bincount(y, minlength=2)))
        res['f1'] = float(f1_score(y, pred, pos_label=minority, zero_division=0))
        res['auc'] = float(roc_auc_score(y, out[:, 1] - out[:, 0]))
    else:
        res['f1'] = float(f1_score(y, pred, average='macro', zero_division=0))
    return {f'{prefix}{k}': v for k, v in res.items()}
