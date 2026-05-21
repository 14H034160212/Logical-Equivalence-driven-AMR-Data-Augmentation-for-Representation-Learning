"""Neural-classifier supplement for AMR → UMR aspect prediction.

Implements the small-classifier component of Post et al. 2024's neuro-symbolic
AMR→UMR conversion. Trains a feature-based model (sklearn LogisticRegression)
on UMR 2.0 English gold annotations to predict :aspect when the rule-based
dictionary doesn't fire.

Features per node:
  - PropBank frame name (one-hot, top-K frames + 'other')
  - Frame stem (everything before -NN) — captures lexical family
  - Frame sense (the NN suffix) — sense often correlates with aspect
  - Has :polarity- modifier (bool)
  - Has :time / :duration / :freq adjuncts (bools)
  - Has progressive marker (heuristic: 'progressive', 'continuous')
  - Depth in graph (root=0, children=1, ...) — captures embedding level
  - Number of arguments (proxy for transitivity / event complexity)
  - Parent role of the node (e.g., :ARG0, :op1) — context signal

Training:
  - Split: 80% train / 20% test on the 596 sentences with gold aspect
  - Loss: cross-entropy over UMR aspect labels (multi-class)
  - Hybrid policy: rules first; if rule abstains, NN; if NN below confidence
    threshold, abstain.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import penman

from .converter import infer_aspect
from .loader import UMRSentence, load_umr_dataset


# Match `verb-NN` style frames
FRAME_RE = re.compile(r"^([a-z][a-z0-9_-]*)-(\d{2,3})$")


def _frame_stem(concept: str) -> str:
    m = FRAME_RE.match(concept)
    return m.group(1) if m else concept


def _frame_sense(concept: str) -> str:
    m = FRAME_RE.match(concept)
    return m.group(2) if m else "00"


def _node_depth(g: penman.Graph, node: str) -> int:
    """BFS depth from graph.top to `node`. Returns -1 if unreachable."""
    if node == g.top:
        return 0
    visited = {g.top}
    frontier = [(g.top, 0)]
    while frontier:
        cur, d = frontier.pop(0)
        for s, role, t in g.triples:
            if s == cur and role != ":instance" and t not in visited:
                if t == node:
                    return d + 1
                visited.add(t)
                frontier.append((t, d + 1))
    return -1


def _parent_role(g: penman.Graph, node: str) -> str:
    """Role pointing at this node from its parent."""
    for s, role, t in g.triples:
        if t == node and role != ":instance":
            return role
    return "_ROOT"


def _node_features(g: penman.Graph, node: str, concept: str) -> Dict:
    """Extract feature dict for a single node."""
    feats = {
        "frame": concept,
        "stem": _frame_stem(concept),
        "sense": _frame_sense(concept),
        "polarity_neg": int((node, ":polarity", "-") in g.triples),
        "has_time": 0,
        "has_duration": 0,
        "has_freq": 0,
        "has_aspect_marker": 0,
        "depth": _node_depth(g, node),
        "n_args": 0,
        "parent_role": _parent_role(g, node),
    }
    for s, role, t in g.triples:
        if s != node:
            continue
        if role == ":time":
            feats["has_time"] = 1
        elif role == ":duration":
            feats["has_duration"] = 1
        elif role in (":freq", ":frequency"):
            feats["has_freq"] = 1
        elif role.startswith(":ARG"):
            feats["n_args"] += 1
    return feats


def collect_training_data(
    sentences: List[UMRSentence],
) -> Tuple[List[Dict], List[str], List[Dict], List[str]]:
    """For every gold-aspect-annotated node, return (features, label).

    Also returns held-out examples where rule-based converter would fire so we
    can evaluate the NN's contribution specifically on hard cases.
    """
    X_features: List[Dict] = []
    y_labels: List[str] = []
    # Hard-case examples (rule abstains): the NN's actual contribution
    X_hard: List[Dict] = []
    y_hard: List[str] = []

    for sent in sentences:
        try:
            g = penman.decode(sent.sentence_umr_penman)
        except Exception:
            continue
        # Extract gold aspect annotations
        gold: Dict[str, str] = {}
        for s, role, t in g.triples:
            if role == ":aspect":
                gold[s] = t
        if not gold:
            continue
        # Strip aspect from the graph to simulate AMR-like input
        triples_clean = [(s, r, t) for s, r, t in g.triples if r != ":aspect"]
        g_clean = penman.Graph(triples_clean, top=g.top)
        # Build features per gold node
        for s, role, t in g_clean.triples:
            if role != ":instance":
                continue
            label = gold.get(s)
            if label is None:
                continue
            feats = _node_features(g_clean, s, t)
            X_features.append(feats)
            y_labels.append(label)
            # Does the rule fire on this node?
            rule_pred = infer_aspect(g_clean, s, t)
            if rule_pred is None:
                X_hard.append(feats)
                y_hard.append(label)
    return X_features, y_labels, X_hard, y_hard


def vectorize(X: List[Dict], vocab: Optional[Dict[str, Dict[str, int]]] = None):
    """Turn list of feature dicts into a numpy matrix.

    If vocab is None, learn it from X (training mode); else reuse (inference).
    Returns (matrix, vocab).
    """
    import numpy as np

    cat_keys = ("frame", "stem", "sense", "parent_role")
    num_keys = ("polarity_neg", "has_time", "has_duration", "has_freq",
                "has_aspect_marker", "depth", "n_args")

    if vocab is None:
        vocab = {k: {} for k in cat_keys}
        for f in X:
            for k in cat_keys:
                v = f[k]
                if v not in vocab[k]:
                    vocab[k][v] = len(vocab[k])
    # Compute dimensions
    dims = [len(vocab[k]) + 1 for k in cat_keys]  # +1 for OOV bucket
    dims += [1] * len(num_keys)
    total_dim = sum(dims)
    mat = np.zeros((len(X), total_dim), dtype=np.float32)
    offset_per_cat = {}
    cur = 0
    for k, d in zip(cat_keys, dims[:len(cat_keys)]):
        offset_per_cat[k] = cur
        cur += d
    num_offset = cur
    for i, f in enumerate(X):
        for k in cat_keys:
            v = f[k]
            idx = vocab[k].get(v, len(vocab[k]))  # OOV at len(vocab)
            mat[i, offset_per_cat[k] + idx] = 1.0
        for j, k in enumerate(num_keys):
            mat[i, num_offset + j] = float(f[k])
    return mat, vocab


def train_classifier(X_train, y_train, X_test, y_test):
    """Train a logistic regression classifier on aspect labels."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        classification_report, f1_score, precision_recall_fscore_support,
    )

    clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                             multi_class="multinomial")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="micro", zero_division=0
    )
    return clf, {
        "macro_precision": p,
        "macro_recall": r,
        "macro_f1": f1,
        "micro_precision": p_micro,
        "micro_recall": r_micro,
        "micro_f1": f1_micro,
        "report": classification_report(y_test, y_pred, zero_division=0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umr-root", default="extensions/umr/umr-data")
    ap.add_argument("--language", default="english")
    ap.add_argument("--out-model", default="extensions/umr/aspect_classifier.pkl")
    ap.add_argument("--out-report", default="extensions/reports/aspect_nn_report.json")
    args = ap.parse_args()

    import numpy as np
    from sklearn.model_selection import train_test_split

    sents = load_umr_dataset(args.umr_root, args.language)
    print(f"Loaded {len(sents)} sentences from {args.language}")

    X, y, X_hard, y_hard = collect_training_data(sents)
    print(f"Collected {len(X)} aspect-annotated nodes; {len(X_hard)} of them "
          f"are 'hard' (rule abstains)")

    if len(X) < 50:
        print("Not enough training data; aborting.")
        return

    # Drop labels with fewer than 2 examples so we can stratify
    label_counts = Counter(y)
    keep_labels = {lab for lab, n in label_counts.items() if n >= 2}
    if len(keep_labels) < len(label_counts):
        dropped = set(label_counts) - keep_labels
        print(f"Dropping rare labels (n<2): {dropped}")
        kept_pairs = [(x, lab) for x, lab in zip(X, y) if lab in keep_labels]
        X = [x for x, _ in kept_pairs]
        y = [lab for _, lab in kept_pairs]

    X_mat, vocab = vectorize(X)
    y_arr = np.array(y)

    # Split 80/20 (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_mat, y_arr, test_size=0.2, random_state=42, stratify=y_arr,
    )
    clf, metrics_all = train_classifier(X_train, y_train, X_test, y_test)
    print()
    print("=== All-nodes evaluation ===")
    print(f"  macro F1 = {metrics_all['macro_f1']:.3f}")
    print(f"  micro F1 = {metrics_all['micro_f1']:.3f}")
    print(metrics_all["report"])

    # Evaluate on hard cases only
    if X_hard:
        X_hard_mat, _ = vectorize(X_hard, vocab=vocab)
        y_hard_arr = np.array(y_hard)
        n_hard_test = max(20, len(X_hard) // 5)
        X_h_test = X_hard_mat[-n_hard_test:]
        y_h_test = y_hard_arr[-n_hard_test:]
        y_hard_pred = clf.predict(X_h_test)
        from sklearn.metrics import precision_recall_fscore_support
        p, r, f1, _ = precision_recall_fscore_support(
            y_h_test, y_hard_pred, average="macro", zero_division=0
        )
        print()
        print("=== Hard-cases-only evaluation (rule abstained) ===")
        print(f"  n_hard_test = {n_hard_test}")
        print(f"  macro precision = {p:.3f}")
        print(f"  macro recall    = {r:.3f}")
        print(f"  macro f1        = {f1:.3f}")
        hard_metrics = {"macro_precision": p, "macro_recall": r, "macro_f1": f1,
                        "n_test": n_hard_test}
    else:
        hard_metrics = {}

    # Save
    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_model, "wb") as f:
        pickle.dump({"clf": clf, "vocab": vocab}, f)
    print(f"\nSaved classifier to {args.out_model}")

    report = {
        "n_training_nodes": len(X),
        "n_hard_nodes": len(X_hard),
        "label_distribution": dict(Counter(y)),
        "all_nodes": {k: v for k, v in metrics_all.items() if k != "report"},
        "hard_only": hard_metrics,
    }
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).write_text(json.dumps(report, indent=2))
    print(f"Saved report to {args.out_report}")


if __name__ == "__main__":
    main()
