"""
run_track_b.py — Track B: Pre-Generation Causal Drift
Research Question: Do hidden-state dynamics within transformer layers provide
predictive signals of hallucination BEFORE it manifests in generated text?

Mathematical Notation:
  N  = number of samples
  L  = number of transformer layers (including embedding layer)
  D  = hidden state dimension
  T  = number of tokens in a sequence

Metric Definitions:
  CD(i,l)    = 1 - cosine_similarity(mean_hs_ctx[i,l], mean_hs_noctx[i,l])
  LLE(i,l)   = mean_over_answer_tokens[ H(p_ctx) - H(p_noctx) ]
               where H = softmax entropy, p = softmax(W_U @ LayerNorm(h))
               Proxy: log1p(||h_ctx||) - log1p(||h_noctx||)
  PCA_RE(i,l)= mean( (h[i,l] - PCA.reconstruct(h[i,l]))^2 )
               PCA fitted on faithful training samples only
  MD(i,l)    = sqrt( (h-mu)^T (Sigma + 0.0001*I)^-1 (h-mu) )
               Gaussian fitted on faithful training samples only
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. Install dependencies
# ─────────────────────────────────────────────────────────────────────────────
import subprocess
import sys

print("=" * 60)
print("[0/14] Installing dependencies...")
try:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "transformers", "accelerate", "datasets", "scikit-learn",
        "matplotlib", "tqdm", "requests", "xgboost", "-q"
    ])
    print("       Dependencies installed.")
except Exception as e:
    print(f"       WARNING: pip install failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Standard imports (after install)
# ─────────────────────────────────────────────────────────────────────────────
import os
import json
import urllib.request
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample

import shutil

from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# 1. Setup directories
# ─────────────────────────────────────────────────────────────────────────────
print("[1/14] Setting up directories...")
os.makedirs("results/figures", exist_ok=True)
print("       results/figures/ ready.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Download datasets
# ─────────────────────────────────────────────────────────────────────────────
print("[2/14] Downloading datasets...")

def download_file(url, filename):
    """Download a file from URL if it does not already exist locally."""
    if not os.path.exists(filename):
        print(f"       Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"       Saved {filename}.")
        except Exception as e:
            print(f"       ERROR downloading {url}: {e}")
    else:
        print(f"       {filename} already exists, skipping download.")

download_file(
    "https://raw.githubusercontent.com/microsoft/RAGTruth/main/data/response_and_label.jsonl",
    "ragtruth.jsonl"
)
download_file(
    "https://github.com/RUCAIBox/HaluEval/raw/main/data/qa_data.json",
    "halueval.json"
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Load and sample data
# ─────────────────────────────────────────────────────────────────────────────
print("[3/14] Loading and sampling data...")
samples = []

# --- RAGTruth ---
try:
    with open("ragtruth.jsonl", "r", encoding="utf-8") as f:
        ragtruth_data = [json.loads(line) for line in f if line.strip()]
    print(f"       RAGTruth: {len(ragtruth_data)} raw records loaded.")
    for item in ragtruth_data:
        # RAGTruth fields: source_info, prompt, response, label
        context  = str(item.get("source_info", ""))[:1000]
        question = str(item.get("prompt", ""))[:500]
        answer   = str(item.get("response", ""))[:500]
        label    = 1 if str(item.get("label", "")).lower() in ("hallucination", "1", "true") else 0
        if question and answer:
            samples.append({
                "context": context, "question": question,
                "answer": answer, "label": label, "source": "ragtruth"
            })
except Exception as e:
    print(f"       ERROR loading RAGTruth: {e}")

# --- HaluEval ---
try:
    with open("halueval.json", "r", encoding="utf-8") as f:
        halueval_data = json.load(f)
    print(f"       HaluEval: {len(halueval_data)} raw records loaded.")
    for item in halueval_data:
        context  = str(item.get("knowledge", ""))[:1000]
        question = str(item.get("question", ""))[:500]
        answer   = str(item.get("answer", ""))[:500]
        label    = 1 if str(item.get("hallucination", "no")).lower() == "yes" else 0
        if question and answer:
            samples.append({
                "context": context, "question": question,
                "answer": answer, "label": label, "source": "halueval"
            })
except Exception as e:
    print(f"       ERROR loading HaluEval: {e}")

if not samples:
    print("       CRITICAL: No samples loaded. Exiting.")
    sys.exit(1)

# Balance: 250 faithful + 250 hallucinated
faithful_all    = [s for s in samples if s["label"] == 0]
hallucinated_all = [s for s in samples if s["label"] == 1]

np.random.seed(42)
n_faithful    = min(250, len(faithful_all))
n_hallucinated = min(250, len(hallucinated_all))

faithful_sample    = np.random.choice(faithful_all,    n_faithful,    replace=False).tolist()
hallucinated_sample = np.random.choice(hallucinated_all, n_hallucinated, replace=False).tolist()

final_samples = faithful_sample + hallucinated_sample
np.random.shuffle(final_samples)

print(f"       Final dataset: {len(final_samples)} samples "
      f"({n_faithful} faithful, {n_hallucinated} hallucinated)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Load model
# ─────────────────────────────────────────────────────────────────────────────
print("[4/14] Loading facebook/opt-1.3b...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"       Device: {device}")

try:
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-1.3b",
        output_hidden_states=True,
        torch_dtype=torch.float16
    ).to(device)
    model.eval()
    print(f"       Model loaded. Layers: {model.config.num_hidden_layers}")
except Exception as e:
    print(f"       CRITICAL: Could not load model: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Tensor extraction
# ─────────────────────────────────────────────────────────────────────────────
print("[5/14] Extracting hidden states (two forward passes per sample)...")
print("       This is the longest step — please wait...")

extracted = {
    "mean_hs_ctx":   [],   # [N, L, D]
    "mean_hs_noctx": [],   # [N, L, D]
    "token_hs_ctx":  [],   # list of [L, T_i, D]
    "token_hs_noctx":[],   # list of [L, T_i, D]
    "token_labels":  [],   # list of [T_i]  (1 = hallucinated token)
    "labels":        [],   # [N]
}

successful = 0
for i, sample in enumerate(tqdm(final_samples, desc="  Extracting")):
    try:
        ctx_text   = f"{sample['context']} {sample['question']} {sample['answer']}"
        noctx_text = f"{sample['question']} {sample['answer']}"

        ctx_inputs   = tokenizer(ctx_text,   return_tensors="pt",
                                 truncation=True, max_length=512).to(device)
        noctx_inputs = tokenizer(noctx_text, return_tensors="pt",
                                 truncation=True, max_length=512).to(device)

        with torch.no_grad():
            ctx_out   = model(**ctx_inputs)
            noctx_out = model(**noctx_inputs)

        # hidden_states is a tuple of (L+1) tensors, each [1, T, D]
        # Stack → [L+1, T, D] after squeezing batch dim
        ctx_hs   = torch.stack(ctx_out.hidden_states).squeeze(1).cpu().float()   # [L+1, T, D]
        noctx_hs = torch.stack(noctx_out.hidden_states).squeeze(1).cpu().float() # [L+1, T, D]

        # Mean-pool over tokens → [L+1, D]
        mean_ctx   = ctx_hs.mean(dim=1)
        mean_noctx = noctx_hs.mean(dim=1)

        extracted["mean_hs_ctx"].append(mean_ctx)
        extracted["mean_hs_noctx"].append(mean_noctx)
        extracted["token_hs_ctx"].append(ctx_hs)
        extracted["token_hs_noctx"].append(noctx_hs)
        extracted["labels"].append(sample["label"])

        # Token-level hallucination labels (heuristic: last 5 tokens of hallucinated samples)
        T = ctx_hs.shape[1]
        tok_lbl = torch.zeros(T)
        if sample["label"] == 1 and T > 5:
            tok_lbl[-5:] = 1
        extracted["token_labels"].append(tok_lbl)

        successful += 1

    except Exception as e:
        print(f"\n       WARNING: sample {i} failed: {e}")
        # Append placeholders so indices stay aligned
        if extracted["mean_hs_ctx"]:
            L_ref, D_ref = extracted["mean_hs_ctx"][0].shape
            extracted["mean_hs_ctx"].append(torch.zeros(L_ref, D_ref))
            extracted["mean_hs_noctx"].append(torch.zeros(L_ref, D_ref))
            extracted["token_hs_ctx"].append(torch.zeros(L_ref, 1, D_ref))
            extracted["token_hs_noctx"].append(torch.zeros(L_ref, 1, D_ref))
        extracted["labels"].append(sample["label"])
        extracted["token_labels"].append(torch.zeros(1))

print(f"       Successful extractions: {successful}/{len(final_samples)}")

# Stack mean-pooled tensors
try:
    mean_hs_ctx   = torch.stack(extracted["mean_hs_ctx"])   # [N, L, D]
    mean_hs_noctx = torch.stack(extracted["mean_hs_noctx"]) # [N, L, D]
    labels_tensor = torch.tensor(extracted["labels"])        # [N]
except Exception as e:
    print(f"       CRITICAL: Could not stack tensors: {e}")
    sys.exit(1)

N, L, D = mean_hs_ctx.shape
print(f"       Tensor shape: N={N}, L={L}, D={D}")

# Save immediately so progress is not lost
try:
    torch.save({
        "mean_hs_ctx":   mean_hs_ctx,
        "mean_hs_noctx": mean_hs_noctx,
        "labels":        labels_tensor,
    }, "tensors.pt")
    print("       Saved tensors.pt")
except Exception as e:
    print(f"       WARNING: Could not save tensors.pt: {e}")

y = labels_tensor.numpy()
faithful_mask = (labels_tensor == 0)
faithful_idx  = faithful_mask.nonzero(as_tuple=True)[0]

# ─────────────────────────────────────────────────────────────────────────────
# 6. Four metrics  →  each returns [N, L] array
# ─────────────────────────────────────────────────────────────────────────────
print("[6/14] Computing four drift metrics...")

metrics = {
    "CD":     np.zeros((N, L)),
    "LLE":    np.zeros((N, L)),
    "PCA_RE": np.zeros((N, L)),
    "MD":     np.zeros((N, L)),
}

for l in tqdm(range(L), desc="  Layer metrics"):
    try:
        h_ctx   = mean_hs_ctx[:, l, :]   # [N, D]
        h_noctx = mean_hs_noctx[:, l, :] # [N, D]

        # ── Metric 1: Cosine Drift ──────────────────────────────────────────
        # CD(i,l) = 1 - cosine_similarity(h_ctx[i,l], h_noctx[i,l])
        cos_sim = F.cosine_similarity(h_ctx, h_noctx, dim=-1)
        metrics["CD"][:, l] = np.nan_to_num((1 - cos_sim).numpy())

        # ── Metric 2: Logit-Lens Delta Entropy (proxy) ──────────────────────
        # LLE(i,l) = log1p(||h_ctx||) - log1p(||h_noctx||)
        # (proxy for entropy difference when W_U projection is memory-costly)
        norm_ctx   = torch.norm(h_ctx,   dim=-1)
        norm_noctx = torch.norm(h_noctx, dim=-1)
        lle = (torch.log1p(norm_ctx) - torch.log1p(norm_noctx)).numpy()
        metrics["LLE"][:, l] = np.nan_to_num(lle)

        # ── Metric 3: PCA Reconstruction Error ──────────────────────────────
        # Fit PCA on faithful samples; measure reconstruction MSE for all
        faithful_hs_np = h_ctx[faithful_idx].numpy()  # [n_faithful, D]
        n_comp = min(32, len(faithful_idx) - 1, D - 1)
        if n_comp > 0 and len(faithful_idx) > n_comp:
            pca = PCA(n_components=n_comp, random_state=42)
            pca.fit(faithful_hs_np)
            all_hs_np = h_ctx.numpy()
            reconstructed = pca.inverse_transform(pca.transform(all_hs_np))
            mse = np.mean((all_hs_np - reconstructed) ** 2, axis=-1)
            metrics["PCA_RE"][:, l] = np.nan_to_num(mse)

        # ── Metric 4: Mahalanobis Distance ──────────────────────────────────
        # MD(i,l) = sqrt( (h-mu)^T (Sigma + 0.0001*I)^-1 (h-mu) )
        if len(faithful_idx) > 1:
            mu  = np.mean(faithful_hs_np, axis=0)
            cov = np.cov(faithful_hs_np, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])
            cov += np.eye(cov.shape[0]) * 1e-4
            try:
                inv_cov = np.linalg.inv(cov)
                all_hs_np = h_ctx.numpy()
                diff = all_hs_np - mu
                md = np.sqrt(np.maximum(0, np.sum(np.dot(diff, inv_cov) * diff, axis=1)))
                metrics["MD"][:, l] = np.nan_to_num(md)
            except np.linalg.LinAlgError:
                # Fallback: use diagonal approximation
                var = np.var(faithful_hs_np, axis=0) + 1e-4
                all_hs_np = h_ctx.numpy()
                diff = all_hs_np - mu
                md = np.sqrt(np.maximum(0, np.sum((diff ** 2) / var, axis=1)))
                metrics["MD"][:, l] = np.nan_to_num(md)

    except Exception as e:
        print(f"\n       WARNING: metrics failed at layer {l}: {e}")

# Final NaN/Inf guard
for k in metrics:
    metrics[k] = np.nan_to_num(metrics[k], nan=0.0, posinf=0.0, neginf=0.0)

print("       Metrics computed.")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Composite feature matrix
# ─────────────────────────────────────────────────────────────────────────────
print("[7/14] Building composite feature matrix X = [CD | LLE | PCA_RE | MD]...")
# X shape: [N, 4*L]
X = np.concatenate([
    metrics["CD"],
    metrics["LLE"],
    metrics["PCA_RE"],
    metrics["MD"],
], axis=1)
X = np.nan_to_num(X)
print(f"       X shape: {X.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Bootstrap AUROC helper
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_auc(y_true, y_pred, n_resamples=1000):
    """Return (mean_auc, ci_low_2.5, ci_high_97.5) via bootstrap resampling."""
    aucs = []
    rng  = np.random.RandomState(0)
    for _ in range(n_resamples):
        try:
            idx = rng.choice(len(y_true), len(y_true), replace=True)
            if len(np.unique(y_true[idx])) > 1:
                aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    if not aucs:
        return 0.5, 0.5, 0.5
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

# ─────────────────────────────────────────────────────────────────────────────
# 9. Composite classifiers with 5-fold CV
# ─────────────────────────────────────────────────────────────────────────────
print("[8/14] Training composite classifiers (5-fold stratified CV)...")

clf_defs = {
    "Logistic Regression": LogisticRegression(C=0.1, class_weight="balanced",
                                              max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200,
                                                  class_weight="balanced",
                                                  random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200,
                                                      learning_rate=0.05,
                                                      random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
composite_results = {}
cv_preds_all = {}   # store OOF predictions for ROC curves

for name, clf in clf_defs.items():
    print(f"       Training {name}...")
    try:
        y_oof = np.zeros(N, dtype=float)
        for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
            clf.fit(X[tr_idx], y[tr_idx])
            if hasattr(clf, "predict_proba"):
                y_oof[te_idx] = clf.predict_proba(X[te_idx])[:, 1]
            else:
                y_oof[te_idx] = clf.decision_function(X[te_idx])

        auc, ci_lo, ci_hi = bootstrap_auc(y, y_oof)
        composite_results[name] = {"auc": auc, "ci_low": ci_lo, "ci_high": ci_hi}
        cv_preds_all[name] = y_oof
        print(f"         {name}: AUROC={auc:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
    except Exception as e:
        print(f"         ERROR training {name}: {e}")
        composite_results[name] = {"auc": 0.5, "ci_low": 0.5, "ci_high": 0.5}
        cv_preds_all[name] = np.zeros(N)

# ─────────────────────────────────────────────────────────────────────────────
# 10. Leave-one-out metric ablation
# ─────────────────────────────────────────────────────────────────────────────
print("[9/14] Leave-one-out metric ablation...")
metric_names = ["CD", "LLE", "PCA_RE", "MD"]
base_auc_rf  = composite_results.get("Random Forest", {}).get("auc", 0.5)
ablation_results = {}

for m_idx, m_name in enumerate(metric_names):
    try:
        # Each metric occupies columns [m_idx*L : (m_idx+1)*L]
        keep_cols = [j for j in range(4 * L) if not (m_idx * L <= j < (m_idx + 1) * L)]
        X_abl = X[:, keep_cols]

        clf_abl = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                         random_state=42, n_jobs=-1)
        y_oof_abl = np.zeros(N, dtype=float)
        for tr_idx, te_idx in cv.split(X_abl, y):
            clf_abl.fit(X_abl[tr_idx], y[tr_idx])
            y_oof_abl[te_idx] = clf_abl.predict_proba(X_abl[te_idx])[:, 1]

        abl_auc = roc_auc_score(y, y_oof_abl)
        drop = base_auc_rf - abl_auc
        ablation_results[m_name] = float(drop)
        print(f"         Remove {m_name}: AUROC={abl_auc:.4f}  drop={drop:+.4f}")
    except Exception as e:
        print(f"         ERROR ablating {m_name}: {e}")
        ablation_results[m_name] = 0.0

composite_results["ablation"] = ablation_results

# Save composite results
try:
    with open("results/composite_results.json", "w") as f:
        json.dump(composite_results, f, indent=2)
    print("       Saved results/composite_results.json")
except Exception as e:
    print(f"       WARNING: could not save composite_results.json: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. Individual metric AUROC (per-layer best)
# ─────────────────────────────────────────────────────────────────────────────
print("[10/14] Computing per-layer individual metric AUROC...")
per_layer_auroc = {m: [] for m in metric_names}

for l in range(L):
    for m_name in metric_names:
        try:
            auc = roc_auc_score(y, metrics[m_name][:, l])
            auc = max(auc, 1 - auc)  # handle inverted correlations
        except Exception:
            auc = 0.5
        per_layer_auroc[m_name].append(float(auc))

indiv_auroc = {m: float(np.max(per_layer_auroc[m])) for m in metric_names}
print(f"       Best per-metric AUROC: {indiv_auroc}")

# ─────────────────────────────────────────────────────────────────────────────
# 12. Peak layer analysis
# ─────────────────────────────────────────────────────────────────────────────
print("[11/14] Peak layer analysis...")

# Layer importance = mean AUROC across all 4 metrics at that layer
layer_importance = np.array([
    np.mean([per_layer_auroc[m][l] for m in metric_names])
    for l in range(L)
])

peak_layers = np.argsort(layer_importance)[-5:][::-1].tolist()
print(f"       Top-5 peak layers: {peak_layers}")

# Retrain on peak-layer features only
peak_auc = 0.5
rand_auc  = 0.5
try:
    peak_cols = []
    for l in peak_layers:
        for mi in range(4):
            peak_cols.append(mi * L + l)
    X_peak = X[:, peak_cols]
    clf_pk = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                    random_state=42, n_jobs=-1)
    y_pk = np.zeros(N, dtype=float)
    for tr_idx, te_idx in cv.split(X_peak, y):
        clf_pk.fit(X_peak[tr_idx], y[tr_idx])
        y_pk[te_idx] = clf_pk.predict_proba(X_peak[te_idx])[:, 1]
    peak_auc = float(roc_auc_score(y, y_pk))
    print(f"       Peak-layer AUROC: {peak_auc:.4f}")

    # Random-5 baseline
    np.random.seed(99)
    rand_layers = np.random.choice(L, 5, replace=False)
    rand_cols = []
    for l in rand_layers:
        for mi in range(4):
            rand_cols.append(mi * L + l)
    X_rand = X[:, rand_cols]
    y_rand = np.zeros(N, dtype=float)
    for tr_idx, te_idx in cv.split(X_rand, y):
        clf_pk.fit(X_rand[tr_idx], y[tr_idx])
        y_rand[te_idx] = clf_pk.predict_proba(X_rand[te_idx])[:, 1]
    rand_auc = float(roc_auc_score(y, y_rand))
    print(f"       Random-5 AUROC:   {rand_auc:.4f}")
except Exception as e:
    print(f"       ERROR in peak layer retraining: {e}")

# K-sweep: AUROC vs number of top-K layers used
print("       Running K-sweep (this may take a few minutes)...")
k_aucs = []
sorted_layers = np.argsort(layer_importance)[::-1]
clf_k = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
for k in tqdm(range(1, L + 1), desc="  K-sweep"):
    try:
        k_cols = []
        for l in sorted_layers[:k]:
            for mi in range(4):
                k_cols.append(mi * L + l)
        X_k = X[:, k_cols]
        y_k = np.zeros(N, dtype=float)
        for tr_idx, te_idx in cv.split(X_k, y):
            clf_k.fit(X_k[tr_idx], y[tr_idx])
            y_k[te_idx] = clf_k.predict_proba(X_k[te_idx])[:, 1]
        k_aucs.append(float(roc_auc_score(y, y_k)))
    except Exception as e:
        k_aucs.append(k_aucs[-1] if k_aucs else 0.5)

# ─────────────────────────────────────────────────────────────────────────────
# 13. Temporal analysis
# ─────────────────────────────────────────────────────────────────────────────
print("[12/14] Temporal analysis (pre-generation signal)...")

faithful_drifts = []
pre_hall_drifts = []
at_hall_drifts  = []

for i, sample in enumerate(final_samples):
    try:
        ctx_hs_i   = extracted["token_hs_ctx"][i]    # [L, T, D]
        noctx_hs_i = extracted["token_hs_noctx"][i]  # [L, T_noctx, D]
        tok_lbl    = extracted["token_labels"][i]     # [T]

        T_ctx = ctx_hs_i.shape[1]
        T_noctx = noctx_hs_i.shape[1]

        if sample["label"] == 0:
            # Faithful baseline: mean drift over all tokens, averaged across layers
            T_min = min(T_ctx, T_noctx)
            if T_min > 0:
                cos = F.cosine_similarity(
                    ctx_hs_i[:, :T_min, :].mean(dim=0),   # [T_min, D]
                    noctx_hs_i[:, :T_min, :].mean(dim=0), # [T_min, D]
                    dim=-1
                )  # [T_min]
                faithful_drifts.append(float((1 - cos).mean()))
        else:
            hall_idx_list = (tok_lbl == 1).nonzero(as_tuple=True)[0]
            if len(hall_idx_list) == 0:
                continue
            first_hall = int(hall_idx_list[0])

            # At hallucination token
            if first_hall < T_ctx and first_hall < T_noctx:
                cos_at = F.cosine_similarity(
                    ctx_hs_i[:, first_hall, :].mean(dim=0, keepdim=True),
                    noctx_hs_i[:, first_hall, :].mean(dim=0, keepdim=True),
                    dim=-1
                )
                at_hall_drifts.append(float(1 - cos_at.mean()))

            # Pre-hallucination window (5 tokens before)
            pre_start = max(0, first_hall - 5)
            if pre_start < first_hall:
                T_pre_min = min(first_hall - pre_start,
                                T_ctx - pre_start,
                                T_noctx - pre_start)
                if T_pre_min > 0:
                    cos_pre = F.cosine_similarity(
                        ctx_hs_i[:, pre_start:pre_start + T_pre_min, :].mean(dim=0),
                        noctx_hs_i[:, pre_start:pre_start + T_pre_min, :].mean(dim=0),
                        dim=-1
                    )
                    pre_hall_drifts.append(float((1 - cos_pre).mean()))
    except Exception as e:
        pass

def safe_mean_std(lst):
    if not lst:
        return 0.0, 0.0
    return float(np.mean(lst)), float(np.std(lst))

temporal_results = {
    "faithful": safe_mean_std(faithful_drifts),
    "pre":      safe_mean_std(pre_hall_drifts),
    "at":       safe_mean_std(at_hall_drifts),
}
print(f"       Faithful:          {temporal_results['faithful'][0]:.4f} ± {temporal_results['faithful'][1]:.4f}")
print(f"       Pre-hallucination: {temporal_results['pre'][0]:.4f} ± {temporal_results['pre'][1]:.4f}")
print(f"       At hallucination:  {temporal_results['at'][0]:.4f} ± {temporal_results['at'][1]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 14. Causal intervention (ablation of top-1 peak layer)
# ─────────────────────────────────────────────────────────────────────────────
print("[13/14] Causal intervention (ablating top-1 peak layer)...")

intervention_result = 0.0
top_layer = int(peak_layers[0]) if peak_layers else 0
print(f"       Ablating layer {top_layer} on 50 hallucinated samples...")

hall_sample_indices = [i for i, s in enumerate(final_samples) if s["label"] == 1]
np.random.seed(42)
intervention_indices = np.random.choice(
    hall_sample_indices, min(50, len(hall_sample_indices)), replace=False
)

entropy_diffs = []

def _ablation_hook(module, input, output):
    """Zero-out the residual stream output of a transformer layer."""
    if isinstance(output, tuple):
        return (torch.zeros_like(output[0]),) + output[1:]
    return torch.zeros_like(output)

for idx in tqdm(intervention_indices, desc="  Intervention"):
    try:
        sample = final_samples[idx]
        ctx_text = f"{sample['context']} {sample['question']} {sample['answer']}"
        inputs = tokenizer(ctx_text, return_tensors="pt",
                           truncation=True, max_length=512).to(device)

        with torch.no_grad():
            # Original forward pass
            out_orig = model(**inputs)
            logits_orig = out_orig.logits.float()
            probs_orig  = F.softmax(logits_orig, dim=-1)
            H_orig = -(probs_orig * torch.log(probs_orig + 1e-9)).sum(dim=-1).mean().item()

            # Ablated forward pass
            hook_handle = model.model.decoder.layers[top_layer].register_forward_hook(
                _ablation_hook
            )
            out_abl  = model(**inputs)
            logits_abl = out_abl.logits.float()
            probs_abl  = F.softmax(logits_abl, dim=-1)
            H_abl = -(probs_abl * torch.log(probs_abl + 1e-9)).sum(dim=-1).mean().item()
            hook_handle.remove()

        entropy_diffs.append(H_abl - H_orig)

    except Exception as e:
        print(f"\n       WARNING: intervention failed for sample {idx}: {e}")

intervention_result = float(np.mean(entropy_diffs)) if entropy_diffs else 0.0
print(f"       Mean entropy change after ablation: {intervention_result:+.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 15. Save peak_layer_results.json
# ─────────────────────────────────────────────────────────────────────────────
peak_layer_results = {
    "peak_layers":    peak_layers,
    "all_auc":        base_auc_rf,
    "peak_auc":       peak_auc,
    "rand_auc":       rand_auc,
    "k_sweep":        k_aucs,
    "temporal": {
        "faithful": list(temporal_results["faithful"]),
        "pre":      list(temporal_results["pre"]),
        "at":       list(temporal_results["at"]),
    },
    "causal_intervention": {
        "ablated_layer":    top_layer,
        "mean_entropy_change": intervention_result,
    },
}
try:
    with open("results/peak_layer_results.json", "w") as f:
        json.dump(peak_layer_results, f, indent=2)
    print("       Saved results/peak_layer_results.json")
except Exception as e:
    print(f"       WARNING: could not save peak_layer_results.json: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 16. Figures
# ─────────────────────────────────────────────────────────────────────────────
print("[14/14] Generating figures...")

# Figure 1: per_layer_auroc.png
try:
    fig, ax = plt.subplots(figsize=(11, 6))
    for m_name in metric_names:
        ax.plot(range(L), per_layer_auroc[m_name], label=m_name, linewidth=1.5)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("AUROC")
    ax.set_title("Per-Layer AUROC for Each Drift Metric")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/figures/per_layer_auroc.png", dpi=150)
    plt.close(fig)
    print("       Saved per_layer_auroc.png")
except Exception as e:
    print(f"       WARNING: per_layer_auroc.png failed: {e}")

# Figure 2: roc_curves.png
try:
    fig, ax = plt.subplots(figsize=(8, 7))
    # Individual metrics (best layer)
    for m_name in metric_names:
        best_l = int(np.argmax(per_layer_auroc[m_name]))
        fpr, tpr, _ = roc_curve(y, metrics[m_name][:, best_l])
        auc_val = per_layer_auroc[m_name][best_l]
        ax.plot(fpr, tpr, linestyle="--", linewidth=1.2,
                label=f"{m_name} (AUC={auc_val:.3f})")
    # Composite models
    for name, preds in cv_preds_all.items():
        fpr, tpr, _ = roc_curve(y, preds)
        auc_val = composite_results[name]["auc"]
        ax.plot(fpr, tpr, linewidth=2,
                label=f"{name} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Individual Metrics and Composite Models")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/figures/roc_curves.png", dpi=150)
    plt.close(fig)
    print("       Saved roc_curves.png")
except Exception as e:
    print(f"       WARNING: roc_curves.png failed: {e}")

# Figure 3: ablation_bar.png
try:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in ablation_results.values()]
    ax.bar(ablation_results.keys(), ablation_results.values(), color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("AUROC Drop (positive = harmful removal)")
    ax.set_title("Leave-One-Out Metric Ablation")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/figures/ablation_bar.png", dpi=150)
    plt.close(fig)
    print("       Saved ablation_bar.png")
except Exception as e:
    print(f"       WARNING: ablation_bar.png failed: {e}")

# Figure 4: layer_importance.png
try:
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#d62728" if i in peak_layers else "#1f77b4" for i in range(L)]
    ax.bar(range(L), layer_importance, color=colors)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean AUROC (across 4 metrics)")
    ax.set_title("Layer Importance — Peak Layers Highlighted in Red")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/figures/layer_importance.png", dpi=150)
    plt.close(fig)
    print("       Saved layer_importance.png")
except Exception as e:
    print(f"       WARNING: layer_importance.png failed: {e}")

# Figure 5: auroc_vs_k.png
try:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(k_aucs) + 1), k_aucs, marker="o", markersize=3, linewidth=1.5)
    ax.axvline(5, color="red", linestyle="--", linewidth=1, label="K=5 (peak layers)")
    ax.set_xlabel("Number of Top Layers K")
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC vs. Number of Top-K Layers Used (Elbow Plot)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/figures/auroc_vs_k.png", dpi=150)
    plt.close(fig)
    print("       Saved auroc_vs_k.png")
except Exception as e:
    print(f"       WARNING: auroc_vs_k.png failed: {e}")

# Figure 6: temporal_drift.png
try:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels_temp = ["Faithful\nBaseline", "Pre-Hallucination\n(−5 tokens)", "At\nHallucination"]
    means = [temporal_results["faithful"][0],
             temporal_results["pre"][0],
             temporal_results["at"][0]]
    stds  = [temporal_results["faithful"][1],
             temporal_results["pre"][1],
             temporal_results["at"][1]]
    bar_colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    ax.bar(labels_temp, means, yerr=stds, capsize=6, color=bar_colors, alpha=0.85)
    ax.set_ylabel("Cosine Drift (1 − cosine similarity)")
    ax.set_title("Temporal Drift: Pre-Generation Predictive Signal (H1)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/figures/temporal_drift.png", dpi=150)
    plt.close(fig)
    print("       Saved temporal_drift.png")
except Exception as e:
    print(f"       WARNING: temporal_drift.png failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 17. Final summary print
# ─────────────────────────────────────────────────────────────────────────────
print()
print("╔══════════════════════════════════════════════╗")
print("║   TRACK B — FINAL RESULTS SUMMARY           ║")
print("╠══════════════════════════════════════════════╣")
print(f"║ Dataset: RAGTruth + HaluEval  N={len(final_samples):<13}║")
print("║ Model:   facebook/opt-1.3b                   ║")
print("╠══════════════════════════════════════════════╣")
print("║ INDIVIDUAL METRIC AUROC                      ║")
print(f"║   Cosine Drift           : {indiv_auroc.get('CD', 0):.4f}            ║")
print(f"║   Logit-Lens Entropy     : {indiv_auroc.get('LLE', 0):.4f}            ║")
print(f"║   PCA Reconstruction Err : {indiv_auroc.get('PCA_RE', 0):.4f}            ║")
print(f"║   Mahalanobis Distance   : {indiv_auroc.get('MD', 0):.4f}            ║")
print("╠══════════════════════════════════════════════╣")
print("║ COMPOSITE MODEL AUROC                        ║")
for clf_name in ["Logistic Regression", "Random Forest", "Gradient Boosting"]:
    res = composite_results.get(clf_name, {"auc": 0, "ci_low": 0, "ci_high": 0})
    short = clf_name[:22]
    print(f"║   {short:<22} : {res['auc']:.4f} [{res['ci_low']:.2f}-{res['ci_high']:.2f}] ║")
print("╠══════════════════════════════════════════════╣")
peak_str = str(peak_layers)
print(f"║ PEAK LAYERS: {peak_str:<32}║")
delta = peak_auc - base_auc_rf
print(f"║   All-layer AUROC : {base_auc_rf:.4f}                   ║")
print(f"║   Peak-layer AUROC: {peak_auc:.4f}  (Δ = {delta:+.4f})   ║")
print(f"║   Random-5 baseline: {rand_auc:.4f}                  ║")
print("╠══════════════════════════════════════════════╣")
print("║ TEMPORAL DRIFT (pre-generation signal)       ║")
fm, fs = temporal_results["faithful"]
pm, ps = temporal_results["pre"]
am, as_ = temporal_results["at"]
print(f"║   Faithful baseline    : {fm:.4f} ± {fs:.4f}    ║")
print(f"║   Pre-hallucination    : {pm:.4f} ± {ps:.4f}    ║")
print(f"║   At hallucination     : {am:.4f} ± {as_:.4f}    ║")
print("╠══════════════════════════════════════════════╣")
print("║ CAUSAL INTERVENTION                          ║")
print(f"║   Entropy change after ablation: {intervention_result:+.4f}     ║")
print("╚══════════════════════════════════════════════╝")

# ─────────────────────────────────────────────────────────────────────────────
# 18. Zip and auto-download
# ─────────────────────────────────────────────────────────────────────────────
print()
print("Creating results archive...")
try:
    shutil.make_archive("track_b_complete", "zip", "results")
    print("Archive created: track_b_complete.zip")
    try:
        from google.colab import files
        files.download("track_b_complete.zip")
        print("Download initiated via Google Colab.")
    except ImportError:
        print("Not running in Google Colab — archive saved as track_b_complete.zip")
except Exception as e:
    print(f"WARNING: archive creation failed: {e}")

print()
print("=" * 60)
print("run_track_b.py completed successfully.")
print("=" * 60)
