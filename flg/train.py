"""Training loop with checkpoint selection on the validation set, and evaluation."""

import copy
import time
from collections import defaultdict

import torch
from torch import nn

from flg.data import encode


def predict(model, examples, alphabet, batch_size=250):
    """Return a list of 0/1 predictions, one per example."""
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            tokens, lengths, _ = encode(examples[start:start + batch_size], alphabet)
            preds += (model(tokens, lengths) > 0).long().tolist()
    return preds


def accuracy(model, examples, alphabet):
    """Accuracy over all examples and separately for each kind (pos, count, order)."""
    preds = predict(model, examples, alphabet)
    correct, total = defaultdict(int), defaultdict(int)
    for pred, (_, label, kind) in zip(preds, examples):
        for key in ("all", kind):
            correct[key] += int(pred == label)
            total[key] += 1
    return {key: correct[key] / total[key] for key in total}


def train_model(model, data, alphabet, cfg, seed):
    """Train on data["train"] and keep the weights with the best accuracy on
    data["val"]. Test data is never used here."""
    torch.manual_seed(seed)
    tokens, lengths, labels = encode(data["train"], alphabet)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)

    best_acc, best_state, best_epoch = -1.0, None, 0
    start = time.time()
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(labels), generator=gen)
        for i in range(0, len(order), cfg["batch_size"]):
            idx = order[i:i + cfg["batch_size"]]
            # Cut padding that no string in this batch needs.
            batch_tokens = tokens[idx, : lengths[idx].max()]
            loss = loss_fn(model(batch_tokens, lengths[idx]), labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_acc = accuracy(model, data["val"], alphabet)["all"]
        if val_acc > best_acc:
            best_acc, best_state, best_epoch = val_acc, copy.deepcopy(model.state_dict()), epoch
        # Stop when validation is perfect (it cannot improve) or has not
        # improved for `patience` epochs.
        if best_acc == 1.0 or epoch - best_epoch >= cfg["patience"]:
            break

    model.load_state_dict(best_state)
    return {
        "val_acc": best_acc,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "seconds": round(time.time() - start, 1),
    }
