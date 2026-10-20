"""Report embeddings: auxiliary training targets for report-grounded supervision.

    python scripts/report_embeddings.py TRAIN_CSV OUT_DIR [--dims 32] [--model intfloat/multilingual-e5-base]

Each training study's free-text report (Spanish) is embedded with a multilingual sentence encoder
(E5: 'passage: ' prefix, attention-masked mean pooling, L2 normalised), then reduced with PCA to
--dims components, scaled so the components' mean variance is 1 (squared error of predicting the
mean is then 1). Readers trained with REPORT_AUX > 0 regress this vector from the images as an
auxiliary task: it carries severity, location and co-findings that the twelve binary labels drop.
Reports are used only as training targets; inference never sees them.

Writes report_embeddings.csv (StudyInstanceUID, rep_0..rep_{dims-1}) and report_embeddings_meta.json.
Needs transformers + tokenizers (not part of the training image; run locally).
"""
import argparse, hashlib, json, os
import numpy as np
import pandas as pd
import torch


def embed(texts, model_name, batch=32):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(['passage: ' + t for t in texts[i:i + batch]], padding=True, truncation=True,
                      max_length=512, return_tensors='pt')
            h = model(**enc).last_hidden_state
            m = enc['attention_mask'].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1)
            out.append(torch.nn.functional.normalize(e, dim=-1).numpy())
    return np.concatenate(out)


def reduce(emb, dims):
    centred = emb - emb.mean(0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    comps = centred @ vt[:dims].T
    scale = np.sqrt(comps.var(0).mean())
    explained = (s[:dims] ** 2).sum() / (s ** 2).sum()
    return comps / scale, float(explained)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('train_csv'); ap.add_argument('out_dir')
    ap.add_argument('--dims', type=int, default=32)
    ap.add_argument('--model', default='intfloat/multilingual-e5-base')
    a = ap.parse_args(argv)
    train = pd.read_csv(a.train_csv, dtype={'StudyInstanceUID': str})
    assert train['StudyInstanceUID'].is_unique and 'Report' in train
    texts = train['Report'].fillna('').astype(str).str.strip().tolist()
    has_text = np.array([len(t) > 0 for t in texts])
    emb = embed([t for t, ok in zip(texts, has_text) if ok], a.model)
    comps, explained = reduce(emb, a.dims)
    table = np.full((len(train), a.dims), np.nan, np.float32)
    table[has_text] = comps
    out = pd.DataFrame(table, columns=[f'rep_{i}' for i in range(a.dims)])
    out.insert(0, 'StudyInstanceUID', train['StudyInstanceUID'].to_numpy())
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, 'report_embeddings.csv')
    out.to_csv(path, index=False, float_format='%.5f')
    meta = dict(model=a.model, dims=a.dims, studies=int(has_text.sum()), empty_reports=int((~has_text).sum()),
                pca_explained=round(explained, 4), sha256=hashlib.sha256(open(path, 'rb').read()).hexdigest())
    with open(os.path.join(a.out_dir, 'report_embeddings_meta.json'), 'w') as handle:
        json.dump(meta, handle, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
