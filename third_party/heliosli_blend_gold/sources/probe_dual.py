"""probe_ controller: split the test studies over the visible GPUs, run probe_infer.py workers, merge.

Never fails the notebook for a single bad study or a dead worker: whatever a worker wrote is used, every
other study gets the per-class mean of the scored studies (0.5 if nothing was scored) and is counted in the
receipt.  Output columns and row order follow sample_submission.csv.
"""
import argparse, csv, json, os, subprocess, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
UIDK = "StudyInstanceUID"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--code", required=True)
    ap.add_argument("--bundle", required=True, help="folder with model.safetensors, pca.npz, feature_norm.npz, heads/")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--amp", default="auto")
    ap.add_argument("--loader-workers", type=int, default=2)
    ap.add_argument("--deadline", type=float, default=0.)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--weights", default=None, help="default: <bundle>/model.safetensors")
    ap.add_argument("--worker", default="probe_infer.py", help="probe_infer.py or compact_infer.py")
    ap.add_argument("--worker-args", default="[]", help="json list of extra worker arguments (compact worker)")
    ap.add_argument("--max-fallback-fraction", type=float, default=0.03)
    a = ap.parse_args()
    t0 = time.time()
    data, out, bundle = Path(a.data), Path(a.output), Path(a.bundle)
    out.mkdir(parents=True, exist_ok=True)
    with open(data / (a.split + ".csv"), encoding="utf-8") as fh:
        ids = [r[UIDK] for r in csv.DictReader(fh)]
    if a.limit:
        ids = ids[:a.limit]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate test ids")
    try:
        names = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                        text=True).strip().splitlines()
    except Exception:
        names = ["unknown"]
    n_gpu = max(1, len(names))

    def cost(uid):                       # number of DICOM files; scheduling only
        n = 0
        try:
            for s in os.scandir(data / (a.split + "_series") / uid):
                if s.is_dir():
                    n += sum(1 for _ in os.scandir(s.path))
        except OSError:
            pass
        return n

    costs = {u: cost(u) for u in ids}
    order = sorted(ids, key=lambda u: (-costs[u], u))
    parts = [order[i::n_gpu] for i in range(n_gpu)]
    # inside a worker: keep the big studies from arriving back to back
    parts = [p[::2] + p[1::2] for p in parts]
    heads_path = str(bundle / "heads.pt") if (bundle / "heads.pt").is_file() else str(bundle / "heads")
    procs = []
    for gi, part in enumerate(parts):
        wdir = out / ("worker%d" % gi)
        wdir.mkdir(exist_ok=True)
        assign = wdir / "ids.json"
        assign.write_text(json.dumps(part), encoding="utf-8")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gi), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MALLOC_ARENA_MAX="2", PYTHONDONTWRITEBYTECODE="1")
        cmd = [sys.executable, "-u", str(HERE / a.worker), "--data", str(data), "--split", a.split,
               "--code", a.code, "--arm", a.arm, "--assignment", str(assign), "--output", str(wdir), "--amp", a.amp,
               "--workers", str(a.loader_workers), "--deadline", str(a.deadline)]
        if a.worker == "probe_infer.py":
            cmd += ["--weights", a.weights or str(bundle / "model.safetensors"), "--pca", str(bundle / "pca.npz"),
                    "--norm", str(bundle / "feature_norm.npz"), "--heads", heads_path]
        else:
            cmd += ["--models", str(bundle)] + [str(x) for x in json.loads(a.worker_args)]
        log = open(wdir / "log.txt", "w", encoding="utf-8")
        procs.append((gi, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log, wdir, part))
        print("WORKER_START", gi, "studies", len(part), "dicom files", sum(costs[u] for u in part), flush=True)
    last = 0
    while any(p.poll() is None for _, p, _, _, _ in procs):
        time.sleep(5)
        if time.time() - last > 60:
            last = time.time()
            for gi, p, _, wdir, part in procs:
                try:
                    tail = [l for l in (wdir / "log.txt").read_text(encoding="utf-8", errors="replace").splitlines()
                            if l.startswith("{")][-1:]
                except OSError:
                    tail = []
                print("PROGRESS worker", gi, "of", len(part), tail[0] if tail else "(starting)", flush=True)
    codes = {}
    for gi, p, log, wdir, part in procs:
        log.close()
        codes[gi] = p.returncode
        if p.returncode != 0:
            print("WORKER_FAILED", gi, "exit", p.returncode, flush=True)
            print((wdir / "log.txt").read_text(encoding="utf-8", errors="replace")[-3000:], flush=True)

    with open(data / "sample_submission.csv", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        sample_ids = [r[0] for r in rd]
    if a.limit:
        sample_ids = [u for u in sample_ids if u in set(ids)]
    if set(sample_ids) != set(ids):
        raise SystemExit("test.csv and sample_submission.csv list different studies")
    labels = header[1:]
    pred, receipts = {}, []
    for gi, p, log, wdir, part in procs:
        f = wdir / "predictions.csv"
        if f.exists():
            with open(f, encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    try:
                        v = [float(r[L]) for L in labels]
                    except (KeyError, ValueError, TypeError):
                        continue
                    if all(0. <= x <= 1. for x in v):
                        pred[r[UIDK]] = v
        try:
            receipts.append(json.loads((wdir / "receipt.json").read_text(encoding="utf-8")))
        except Exception:
            receipts.append({"status": "NO_RECEIPT", "worker": gi})
    scored = [u for u in sample_ids if u in pred]
    if scored:
        mean = [sum(pred[u][j] for u in scored) / len(scored) for j in range(len(labels))]
    else:
        mean = [0.5] * len(labels)
    missing = [u for u in sample_ids if u not in pred]
    with open(out / "submission.csv", "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        for u in sample_ids:
            wr.writerow([u] + ["%.6f" % x for x in pred.get(u, mean)])
    reasons = {}
    for r in receipts:
        for u, why in (r.get("fallback") or {}).items():
            reasons[u] = why
    total = {"decode_s": 0., "render_s": 0., "gpu_s": 0., "windows": 0., "rows": 0.}
    for r in receipts:
        for k in total:
            total[k] += float((r.get("stats") or {}).get(k, 0.))
    n = max(1., total["rows"])
    result = {"status": "PASS" if scored else "ALL_FALLBACK", "arm": a.arm, "rows": len(sample_ids),
              "scored": len(scored), "fallback_studies": len(missing),
              "fallback_reasons": {u: reasons.get(u, "worker did not report") for u in missing[:40]},
              "worker_exit_codes": codes, "gpus": names, "seconds": round(time.time() - t0, 1),
              "per_study_mean": {"decode_s": round(total["decode_s"] / n, 3), "render_s": round(total["render_s"] / n, 3),
                                 "gpu_s": round(total["gpu_s"] / n, 3), "windows": round(total["windows"] / n, 1)},
              "workers": receipts, "worker": a.worker}
    (out / "submission_receipt.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    if len(missing) > a.max_fallback_fraction * len(sample_ids):
        raise SystemExit("PROBE_TOO_MANY_FALLBACKS %d of %d (limit %.0f%%); see submission_receipt.json" % (
            len(missing), len(sample_ids), 100 * a.max_fallback_fraction))
    print("CONTROLLER_DONE", json.dumps({k: result[k] for k in ("status", "rows", "scored", "fallback_studies",
                                                                "seconds", "per_study_mean", "worker_exit_codes")}), flush=True)


if __name__ == "__main__":
    main()
