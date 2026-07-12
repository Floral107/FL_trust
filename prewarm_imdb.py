#!/usr/bin/env python3
"""prewarm_imdb.py K — embed all imdb (partition,K) train+test splits ONCE so the
content-hash cache is warm; every later cell (all seeds/metrics/modes) then loads
embeddings from disk (bit-identical) instead of re-running USE. Run one process
per K (fresh FederatedDataset): prewarm_imdb.py 1 (server eval set), 4, 20.
"""
import os, sys, time
os.environ.setdefault("seed", "42")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("HF_DATASETS_OFFLINE", "0")
sys.path.insert(0, os.getcwd())
import imdbnoniid.imdbnoniid.task as T

def main():
    K = int(sys.argv[1])
    t0 = time.time()
    for p in range(K):
        xtr, ytr, xte, yte = T.load_data(p, K)
        T.process_text(xtr); T.process_text(xte)
        print(f"[prewarm-imdb] K={K} p={p}: {len(xtr)}tr+{len(xte)}te cached ({time.time()-t0:.0f}s)", flush=True)
    print(f"[prewarm-imdb] K={K} COMPLETE ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
