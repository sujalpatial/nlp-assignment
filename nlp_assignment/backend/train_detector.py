"""
train_detector.py — Project Sentinel
Trains the GradientBoosting hallucination detector.
Run once: python train_detector.py
Output:   data/detector.pkl
"""

import os, json, pickle, logging, warnings, requests
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.ensemble        import GradientBoostingClassifier
from sklearn.decomposition   import PCA
from sklearn.covariance      import LedoitWolf
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics         import roc_auc_score
from transformers            import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME   = "facebook/opt-1.3b"
N_SAMPLES    = 500
MAX_ANS_LEN  = 64
MAX_CTX_LEN  = 256
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE        = torch.float16 if torch.cuda.is_available() else torch.float32
PCA_COMPS    = 16
TOP_K_LAYERS = 5
DATA_DIR     = "data"
OUT_PATH     = os.path.join(DATA_DIR, "detector.pkl")
os.makedirs(DATA_DIR, exist_ok=True)


def _synthetic_samples(n, faithful):
    templates_f = [
        ("The capital of France is Paris.", "What is the capital of France?"),
        ("Water boils at 100 degrees Celsius.", "At what temperature does water boil?"),
        ("Einstein was born in 1879.", "When was Einstein born?"),
        ("The speed of light is 299792458 m/s.", "What is the speed of light?"),
        ("DNA carries genetic information.", "What does DNA do?"),
    ]
    templates_h = [
        ("The capital of France is London.", "What is the capital of France?"),
        ("Water boils at 50 degrees Celsius.", "At what temperature does water boil?"),
        ("Einstein was born in 1920.", "When was Einstein born?"),
        ("The speed of light is 100000 m/s.", "What is the speed of light?"),
        ("DNA carries electrical signals.", "What does DNA do?"),
    ]
    templates = templates_f if faithful else templates_h
    samples = []
    for i in range(n):
        ans, q = templates[i % len(templates)]
        samples.append({
            "context": f"Reference: {ans}", "question": q, "answer": ans,
            "label": 0 if faithful else 1,
            "hallucination_spans": [] if faithful else [(0, len(ans))],
        })
    return samples


def load_ragtruth(max_n=300):
    url  = "https://raw.githubusercontent.com/microsoft/RAGTruth/main/data/response_and_label.jsonl"
    path = os.path.join(DATA_DIR, "ragtruth.jsonl")
    if not os.path.exists(path):
        try:
            r = requests.get(url, timeout=30); r.raise_for_status()
            open(path, "wb").write(r.content)
        except Exception as e:
            log.warning("RAGTruth download failed: %s", e); return []
    samples = []
    try:
        with open(path) as f:
            for line in f:
                obj  = json.loads(line.strip())
                src  = obj.get("source_info", obj)
                q    = src.get("question", "")
                docs = src.get("passages", src.get("docs", []))
                ctx  = " ".join([d.get("passage", d.get("text", str(d))) for d in docs])[:600]
                ans  = obj.get("model_response", obj.get("response", ""))
                lbls = obj.get("labels", [])
                spans = [(l["start"], l["end"]) for l in lbls
                         if l.get("label","").lower() in ("hallucination","intrinsic","extrinsic")]
                if not q or not ans: continue
                samples.append({"context": ctx, "question": q, "answer": ans,
                                 "label": 1 if spans else 0, "hallucination_spans": spans})
                if len(samples) >= max_n: break
    except Exception as e:
        log.warning("RAGTruth parse error: %s", e)
    log.info("RAGTruth: %d samples", len(samples))
    return samples


def load_halueval(max_n=300):
    url  = "https://github.com/RUCAIBox/HaluEval/raw/main/data/qa_data.json"
    path = os.path.join(DATA_DIR, "halueval_qa.json")
    if not os.path.exists(path):
        try:
            r = requests.get(url, timeout=30); r.raise_for_status()
            open(path, "wb").write(r.content)
        except Exception as e:
            log.warning("HaluEval download failed: %s", e); return []
    try:
        data = json.load(open(path))
        samples = []
        for obj in data:
            q, ctx = obj.get("question",""), obj.get("knowledge","")[:600]
            ra, ha = obj.get("right_answer",""), obj.get("hallucinated_answer","")
            if not q: continue
            if ra: samples.append({"context":ctx,"question":q,"answer":ra,"label":0,"hallucination_spans":[]})
            if ha: samples.append({"context":ctx,"question":q,"answer":ha,"label":1,"hallucination_spans":[(0,len(ha))]})
            if len(samples) >= max_n: break
        log.info("HaluEval: %d samples", len(samples))
        return samples
    except Exception as e:
        log.warning("HaluEval parse error: %s", e); return []


def build_dataset():
    import random
    rt  = load_ragtruth(300)
    hev = load_halueval(300)
    all_s = rt + hev
    random.seed(42); random.shuffle(all_s)
    faithful     = [s for s in all_s if s["label"] == 0]
    hallucinated = [s for s in all_s if s["label"] == 1]
    half = N_SAMPLES // 2
    while len(faithful)     < half: faithful     += _synthetic_samples(half, True)
    while len(hallucinated) < half: hallucinated += _synthetic_samples(half, False)
    combined = faithful[:half] + hallucinated[:half]
    random.shuffle(combined)
    log.info("Dataset: %d total (%d faithful, %d hallucinated)", len(combined), half, half)
    return combined


def load_model():
    log.info("Loading %s on %s...", MODEL_NAME, DEVICE)
    tok   = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE,
        device_map="auto" if DEVICE=="cuda" else None,
        output_hidden_states=True,
    )
    model.eval()
    if DEVICE == "cpu": model = model.to(DEVICE)
    log.info("Loaded: layers=%d hidden=%d", model.config.num_hidden_layers, model.config.hidden_size)
    return model, tok, model.config.num_hidden_layers, model.config.hidden_size


def _encode(tok, text, max_len):
    ids = tok.encode(text, add_special_tokens=False)
    return torch.tensor(ids[:max_len], dtype=torch.long)


@torch.no_grad()
def _forward(model, ids, ans_len):
    out = model(ids.unsqueeze(0).to(DEVICE), output_hidden_states=True)
    hs  = torch.stack(out.hidden_states[1:], dim=0).squeeze(1).float()
    ans_start = max(hs.shape[1] - ans_len, 0)
    return hs[:, ans_start:, :].mean(dim=1).cpu()


def extract_tensors(samples, model, tok, n_layers, hidden_dim):
    N = len(samples)
    ctx_arr   = np.zeros((N, n_layers, hidden_dim), dtype=np.float32)
    noctx_arr = np.zeros((N, n_layers, hidden_dim), dtype=np.float32)
    labels    = np.zeros(N, dtype=np.int32)
    for i, s in enumerate(tqdm(samples, desc="Extracting")):
        try:
            ctx_ids   = _encode(tok, s["context"][:400] + "\nQ: " + s["question"] + "\nA:", MAX_CTX_LEN)
            noctx_ids = _encode(tok, "Q: " + s["question"] + "\nA:", 80)
            ans_ids   = _encode(tok, s["answer"], MAX_ANS_LEN)
            ans_len   = max(len(ans_ids), 1)
            bos = torch.tensor([tok.bos_token_id], dtype=torch.long) if tok.bos_token_id else torch.tensor([], dtype=torch.long)
            ctx_arr[i]   = _forward(model, torch.cat([bos, ctx_ids,   ans_ids]), ans_len).numpy()
            noctx_arr[i] = _forward(model, torch.cat([bos, noctx_ids, ans_ids]), ans_len).numpy()
            labels[i]    = s["label"]
        except Exception as e:
            log.warning("Sample %d skipped: %s", i, e)
    return ctx_arr, noctx_arr, labels


def cosine_drift(ctx, noctx):
    n = np.linalg.norm(ctx,   axis=-1, keepdims=True) + 1e-8
    m = np.linalg.norm(noctx, axis=-1, keepdims=True) + 1e-8
    return np.clip(1 - np.sum((ctx/n)*(noctx/m), axis=-1), 0, 2)


def logit_lens_entropy(ctx, noctx, model):
    W = model.get_output_embeddings().weight.detach().cpu().float().numpy()
    N, L, D = ctx.shape
    delta = np.zeros((N, L), dtype=np.float32)
    for l in range(L):
        for arr, sign in [(ctx, 1), (noctx, -1)]:
            logits = arr[:, l, :] @ W.T
            logits -= logits.max(axis=-1, keepdims=True)
            probs   = np.exp(logits); probs /= probs.sum(axis=-1, keepdims=True)
            ent     = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
            delta[:, l] += sign * ent
    return delta


def pca_re(ctx, labels, train_mask):
    N, L, D = ctx.shape; X = np.zeros((N, L), dtype=np.float32)
    for l in range(L):
        h_f = ctx[train_mask & (labels==0), l, :]
        if len(h_f) < PCA_COMPS+1: continue
        k   = min(PCA_COMPS, len(h_f)-1, D-1)
        pca = PCA(n_components=k, random_state=42).fit(h_f)
        h   = ctx[:, l, :]
        X[:, l] = np.mean((h - pca.inverse_transform(pca.transform(h)))**2, axis=-1)
    return X


def mahal(ctx, labels, train_mask):
    N, L, D = ctx.shape; X = np.zeros((N, L), dtype=np.float32)
    for l in range(L):
        h_f = ctx[train_mask & (labels==0), l, :]
        if len(h_f) < 5: continue
        try:
            lw      = LedoitWolf().fit(h_f)
            mu      = h_f.mean(0)
            cov_inv = np.linalg.inv(lw.covariance_ + 1e-5*np.eye(D))
            diff    = ctx[:, l, :] - mu
            X[:, l] = np.sqrt(np.clip(np.einsum("ni,ij,nj->n", diff, cov_inv, diff), 0, None))
        except Exception:
            pass
    return X


def main():
    log.info("="*60)
    log.info("Project Sentinel — Training Detector")
    log.info("="*60)

    samples = build_dataset()
    y       = np.array([s["label"] for s in samples])
    model, tok, n_layers, hidden_dim = load_model()

    log.info("Extracting hidden states...")
    ctx, noctx, y = extract_tensors(samples, model, tok, n_layers, hidden_dim)

    rng        = np.random.default_rng(42)
    idx        = np.arange(len(y)); rng.shuffle(idx)
    train_mask = np.zeros(len(y), dtype=bool)
    train_mask[idx[:int(len(y)*0.75)]] = True

    log.info("Computing metrics...")
    metrics = {
        "cosine_drift" : np.nan_to_num(cosine_drift(ctx, noctx)),
        "logit_lens"   : np.nan_to_num(logit_lens_entropy(ctx, noctx, model)),
        "pca_re"       : np.nan_to_num(pca_re(ctx, y, train_mask)),
        "mahalanobis"  : np.nan_to_num(mahal(ctx, y, train_mask)),
    }

    # Peak layers
    L          = n_layers
    importance = np.zeros(L)
    for arr in metrics.values():
        for l in range(L):
            s = arr[:, l]
            if np.std(s) < 1e-8: continue
            auc = roc_auc_score(y, s)
            importance[l] += max(auc, 1-auc)
    importance /= len(metrics)
    peak_layers = sorted(np.argsort(importance)[::-1][:TOP_K_LAYERS].tolist())
    log.info("Peak layers: %s", peak_layers)

    # Feature matrix
    X = np.nan_to_num(np.concatenate([metrics[n][:, peak_layers] for n in metrics], axis=1))

    log.info("Training GradientBoostingClassifier...")
    gbc   = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                        learning_rate=0.05, subsample=0.8, random_state=42)
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = cross_val_predict(gbc, X, y, cv=cv, method="predict_proba")[:, 1]
    auroc = roc_auc_score(y, probs)
    log.info("5-fold CV AUROC = %.4f", auroc)
    gbc.fit(X, y)

    # Per-layer manifold
    pca_models, manifold_means, manifold_covs, faithful_mean_hs = {}, {}, {}, {}
    for l in peak_layers:
        h_f = ctx[y==0, l, :]
        if len(h_f) < PCA_COMPS+1: continue
        k = min(PCA_COMPS, len(h_f)-1, hidden_dim-1)
        pca_models[l]     = PCA(n_components=k, random_state=42).fit(h_f)
        lw                = LedoitWolf().fit(h_f)
        manifold_means[l] = h_f.mean(0)
        manifold_covs[l]  = lw.covariance_
        faithful_mean_hs[l] = h_f

    payload = {
        "gbc": gbc, "pca_models": pca_models,
        "manifold_means": manifold_means, "manifold_covs": manifold_covs,
        "faithful_mean_hs": faithful_mean_hs,
        "peak_layers": peak_layers, "layer_importance": importance.tolist(),
        "auroc": auroc, "n_layers": n_layers, "hidden_dim": hidden_dim,
    }
    with open(OUT_PATH, "wb") as f:
        pickle.dump(payload, f)

    print("\n" + "="*56)
    print("  DETECTOR TRAINING COMPLETE")
    print("="*56)
    print(f"  Samples : {len(y)} ({(y==0).sum()} faithful, {(y==1).sum()} hallucinated)")
    print(f"  AUROC   : {auroc:.4f}")
    print(f"  Peaks   : {peak_layers}")
    print(f"  Output  : {OUT_PATH}")
    print("="*56)

if __name__ == "__main__":
    main()
