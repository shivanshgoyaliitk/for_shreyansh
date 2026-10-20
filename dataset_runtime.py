"""Training dataset, embedded in training.ipynb after the fold helpers."""
import cv2

class KneeDS(Dataset):
    def __init__(self, df, rows, train):
        self.df = df.reset_index(drop=True)
        self.rows = rows
        self.train = train
        self.y = self.df[LABEL_COLS].to_numpy(np.float32)
        # Per-cell loss weights (gold rows are up-weighted); unweighted tables default to 1.
        weight_cols = [f'w_{c}' for c in LABEL_COLS]
        self.w = (self.df[weight_cols].to_numpy(np.float32) if set(weight_cols) <= set(self.df)
                  else np.ones_like(self.y))
        self.present = self.df[[f"present_{s}" for s in range(len(PLANES))]].to_numpy(bool)
        self.row_of = self.df["row"].to_numpy(int)
        self.shard_of = (self.df["cache_id"].to_numpy(int) if "cache_id" in self.df
                         else np.zeros(len(self.df), int))
        self.caches = {}                       # memmaps opened lazily, per worker and shard
        # Training batches carry the report-embedding targets after the labels (weight 1 where known).
        rep = globals().get('REP_COLS') or []
        self.aux = self.df[rep].to_numpy(np.float32) if (train and rep) else None

    def _cache(self, shard):
        if shard not in self.caches:
            self.caches[shard] = np.load(os.path.join(CACHE_DIRS[shard], "cache.npy"), mmap_mode="r")
        return self.caches[shard]

    def __len__(self):
        return len(self.rows)

    def _aug(self, vol):
        # NO horizontal or vertical flip: a horizontal flip in the coronal plane swaps
        # medial and lateral, which would invert four of the twelve labels.
        h, w = vol.shape[-2:]
        ang = np.random.uniform(-12, 12)
        sc  = np.random.uniform(0.9, 1.1)
        M = cv2.getRotationMatrix2D((w/2, h/2), ang, sc)
        M[:, 2] += np.random.uniform(-0.06, 0.06, 2) * w
        out = np.empty_like(vol)
        for i in range(vol.shape[0]):
            out[i] = cv2.warpAffine(vol[i], M, (w, h), flags=cv2.INTER_LINEAR,
                                    borderValue=0)
        if np.random.rand() < 0.5:
            out = np.clip(out.astype(np.float32) * np.random.uniform(0.85, 1.15)
                          + np.random.uniform(-15, 15), 0, 255).astype(np.uint8)
        return out

    def _aug_slot(self, vol):
        # Public-member jitter for slot_query: small rotation, zoom-in only (zooming out
        # would invent tissue at the crop edge, where the popliteal fossa sits), edge
        # replication, multiplicative intensity. Still no flips.
        h, w = vol.shape[-2:]
        M = cv2.getRotationMatrix2D((w/2, h/2), np.random.uniform(-8, 8),
                                    np.random.uniform(1.0, 1.08))
        M[:, 2] += np.random.uniform(-0.025, 0.025, 2) * w
        out = np.stack([cv2.warpAffine(s, M, (w, h), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE) for s in vol])
        return np.clip(out.astype(np.float32) * np.random.uniform(0.9, 1.1),
                       0, 255).astype(np.uint8)

    def __getitem__(self, i):
        j = self.rows[i]
        r = self.row_of[j]
        cache = self._cache(self.shard_of[j])
        present = self.present[j]

        if self.train and ARCH == 'reader25d':
            # All windows; the model subsamples windows and augments each series on GPU.
            vol = np.asarray(cache[r])
        elif self.train and ARCH == 'slot_query':
            # One slice from each third of the band, drawn independently per slot, so
            # every cached position trains while each input still spans the joint.
            parts = slice_thirds(cache.shape[2])
            vol = np.stack([self._aug_slot(np.asarray(cache[r, s, [np.random.choice(p) for p in parts]]))
                            if present[s] else np.zeros((3, *cache.shape[3:]), np.uint8)
                            for s in range(cache.shape[1])])
        elif self.train:
            # Slice the memmap BEFORE materialising, so the discarded slices are
            # never read off disk at all (~37% less I/O per sample).
            avail = cache.shape[2]
            t = min(T_TRAIN, avail)
            t0 = np.random.randint(0, avail - t + 1)
            vol = np.asarray(cache[r][:, t0:t0 + t])
            vol = np.stack([self._aug(v) if present[s] else v
                            for s, v in enumerate(vol)])
        else:
            vol = np.asarray(cache[r])               # eval uses every slice

        y, w = self.y[j], self.w[j]
        if self.aux is not None:
            y = np.concatenate([y, self.aux[j]])
            w = np.concatenate([w, np.isfinite(self.aux[j]).astype(np.float32)])
        return (torch.from_numpy(np.array(vol, copy=True, order='C')),
                torch.from_numpy(present.copy()),
                torch.from_numpy(y.copy()),
                torch.from_numpy(w.copy()))


def collate(batch):
    xs, ps, ys, ws = zip(*batch)
    return torch.stack(xs), torch.stack(ps), torch.stack(ys), torch.stack(ws)
