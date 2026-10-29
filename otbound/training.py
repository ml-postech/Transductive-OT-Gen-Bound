"""Shared training loop for the transductive node classification experiments.

All three graph experiments -- correlation (Table 1), regularization (Table 2) and depth
(Figure 1) -- train the same way and differ only in what they measure afterwards and
whether a regularizer is attached.  That loop lives here so the experiment scripts stay
about measurement.

Training is full-batch: the graph is small enough to forward in one pass, and message
passing makes node representations interdependent anyway, so mini-batching would change
the model rather than just the optimizer.
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["TrainHistory", "train_graph_model", "evaluate_graph_model", "task_loss",
           "train_tabular_model", "evaluate_tabular_model", "tabular_task_loss"]


class TrainHistory(object):
    """Per-epoch curves and the best validation score seen.

    ``train_acc`` and ``train_loss`` are what Appendix F.1 plots to show that
    Amazon-ratings fails to fit its training set, the regime where regularization is not
    expected to help.
    """

    def __init__(self):
        self.train_loss = []
        self.train_acc = []
        self.val_acc = []
        self.test_acc = []
        self.best_val_acc = 0.0
        self.best_test_acc = 0.0
        self.epochs_run = 0


def task_loss(logits, y, mask):
    """Cross entropy on ``mask``; binary tasks use the logistic loss on one logit."""
    if int(y.max()) == 1:
        return F.binary_cross_entropy_with_logits(logits[mask], y[mask].float())
    return F.cross_entropy(logits[mask], y[mask])


@torch.no_grad()
def evaluate_graph_model(model, data, masks):
    """Accuracy and loss on each mask in ``masks`` (``None`` entries are skipped)."""
    model.eval()
    out = model(data.x, data.edge_index)
    binary = int(data.y.max()) == 1
    pred = (out.sigmoid() > 0.5).long() if binary else out.argmax(dim=-1)

    accs, losses = [], []
    for mask in masks:
        if mask is None:
            continue
        accs.append(float((pred[mask] == data.y[mask]).sum()) / float(mask.sum()))
        losses.append(float(task_loss(out, data.y, mask)))
    return accs, losses


def train_graph_model(model, data, args, device, regularizer_fn=None, verbose=False):
    """Train one model to completion.

    Args:
        model: something exposing ``forward(x, edge_index)`` and ``feature(x, edge_index)``.
        data: PyG ``Data`` with ``train_mask`` / ``test_mask`` and optionally ``val_mask``.
        args: parsed arguments; reads ``lr``, ``weight_decay``, ``epochs``,
            ``early_stopping`` and ``use_val``.
        regularizer_fn: optional ``fn(logits, features, data) -> scalar`` added to the
            task loss.  Section 5.2 passes the OT regularizer here; ``None`` gives the
            unregularized baseline of the same architecture.
        verbose: print a per-epoch line.

    Returns:
        ``(model, history)``.  With a validation set the returned model is restored to
        the best-validation checkpoint, which is what the reported test accuracy refers
        to; without one, it is the model at the last epoch.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    use_val = bool(getattr(args, "use_val", False)) and getattr(data, "val_mask", None) is not None
    history = TrainHistory()
    best_state = None
    patience = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = task_loss(out, data.y, data.train_mask)

        if regularizer_fn is not None:
            # A second encoder pass: the regularizer needs phi(x), while the task loss
            # needs the logits.  Cheaper than caching the whole graph across the two.
            features = model.feature(data.x, data.edge_index)
            loss = loss + regularizer_fn(out, features, data)

        loss.backward()
        optimizer.step()

        masks = [data.train_mask, data.val_mask if use_val else None, data.test_mask]
        accs, losses = evaluate_graph_model(model, data, masks)
        if use_val:
            train_acc, val_acc, test_acc = accs
        else:
            train_acc, test_acc = accs
            val_acc = None

        history.train_loss.append(losses[0])
        history.train_acc.append(train_acc)
        history.test_acc.append(test_acc)
        history.epochs_run = epoch

        if use_val:
            history.val_acc.append(val_acc)
            if val_acc > history.best_val_acc:
                history.best_val_acc = val_acc
                history.best_test_acc = test_acc
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
        else:
            history.best_test_acc = test_acc

        if verbose:
            print("epoch {:4d}  loss {:.4f}  train {:.4f}  test {:.4f}{}".format(
                epoch, losses[0], train_acc, test_acc,
                "  val {:.4f}".format(val_acc) if use_val else ""))

        if args.early_stopping > 0 and patience >= args.early_stopping:
            break
        if np.isnan(losses[0]):
            print("training diverged (NaN loss) at epoch {}; stopping".format(epoch))
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def tabular_task_loss(logits, y):
    """Cross entropy, or the logistic loss when the head emits a single logit."""
    if logits.dim() == 1:
        return F.binary_cross_entropy_with_logits(logits, y.float())
    return F.cross_entropy(logits, y)


@torch.no_grad()
def evaluate_tabular_model(model, X, y, masks, batch_size=0):
    """Accuracy on each mask, forwarding in chunks when ``batch_size > 0``."""
    from .tabular import batched_forward

    model.eval()
    out = batched_forward(model, X, batch_size)
    pred = (out.sigmoid() > 0.5).long() if out.dim() == 1 else out.argmax(-1)
    return [float((pred[m] == y[m]).sum()) / float(m.sum()) for m in masks if m is not None]


def train_tabular_model(model, X, y, train_mask, test_mask, args, device,
                        val_mask=None, regularizer_fn=None, verbose=False):
    """Train a tabular or image model, full-batch or mini-batch.

    ``args.batch_size == 0`` keeps the whole pool in one step, which is how the graph
    experiments run and is possible for the small tabular datasets.  Otherwise labelled
    points are shuffled into mini-batches and, when a regularizer is attached, a
    proportionally sized unlabelled batch is drawn fresh each step so that :math:`M_c`
    still sees unlabelled features.

    ``regularizer_fn`` has signature
    ``fn(train_features, train_logits, train_y, test_features, test_logits) -> scalar``.

    Returns ``(model, history)``; with a validation mask the best-validation state is
    restored.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    batch_size = getattr(args, "batch_size", 0)

    train_idx = train_mask.nonzero(as_tuple=True)[0]
    test_idx = test_mask.nonzero(as_tuple=True)[0]
    n_train, n_test = len(train_idx), len(test_idx)
    # Keep the labelled/unlabelled ratio of the step equal to the ratio of the pool.
    test_per_step = max(1, int(round(batch_size * n_test / max(n_train, 1)))) if batch_size else n_test

    history = TrainHistory()
    best_state = None
    patience = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        if batch_size == 0:
            steps = [(train_idx, test_idx)]
        else:
            shuffled = train_idx[torch.randperm(n_train, device=train_idx.device)]
            steps = []
            for i in range(0, n_train, batch_size):
                tb = shuffled[i:i + batch_size]
                te = test_idx[torch.randperm(n_test, device=test_idx.device)[:test_per_step]]
                steps.append((tb, te))

        for tb_idx, te_idx in steps:
            optimizer.zero_grad()
            train_logits = model(X[tb_idx])
            loss = tabular_task_loss(train_logits, y[tb_idx])

            if regularizer_fn is not None:
                train_features = model.feature(X[tb_idx])
                test_logits = model(X[te_idx])
                test_features = model.feature(X[te_idx])
                loss = loss + regularizer_fn(
                    train_features, train_logits, y[tb_idx],
                    test_features, test_logits,
                )

            loss.backward()
            optimizer.step()

        masks = [train_mask, val_mask, test_mask]
        accs = evaluate_tabular_model(model, X, y, masks, batch_size)
        if val_mask is not None:
            train_acc, val_acc, test_acc = accs
        else:
            train_acc, test_acc = accs
            val_acc = None

        history.train_loss.append(float(loss))
        history.train_acc.append(train_acc)
        history.test_acc.append(test_acc)
        history.epochs_run = epoch

        if val_mask is not None:
            history.val_acc.append(val_acc)
            if val_acc > history.best_val_acc:
                history.best_val_acc = val_acc
                history.best_test_acc = test_acc
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
        else:
            history.best_test_acc = test_acc

        if verbose:
            print("epoch {:4d}  loss {:.4f}  train {:.4f}  test {:.4f}".format(
                epoch, float(loss), train_acc, test_acc))

        if args.early_stopping > 0 and patience >= args.early_stopping:
            break
        if np.isnan(float(loss)):
            print("training diverged (NaN loss) at epoch {}; stopping".format(epoch))
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
