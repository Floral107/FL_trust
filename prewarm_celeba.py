#!/usr/bin/env python3
"""prewarm_celeba.py K — serially build all celeba (partition,K) .npz caches.

The celeba prep (HF load + PIL decode + compress) is guarded by a single
dataset-wide lock. When many celeba cells start on a COLD cache, their ~K client
actors thrash that one lock and nothing reaches GPU scoring. This script does the
one-time cold prep ONCE, serially, in a single process (no Ray, no contention),
so that every later cell hits the fast np.load() path.

Run ONE process per K (a fresh FederatedDataset each, since the module-global
`fds` is bound to its num_partitions):
    CUDA_VISIBLE_DEVICES= python3 prewarm_celeba.py 4
    CUDA_VISIBLE_DEVICES= python3 prewarm_celeba.py 20
Idempotent: partitions already cached are skipped.
"""
import os, sys, time
sys.path.insert(0, os.getcwd())
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("HF_DATASETS_OFFLINE", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # prep is CPU-only
os.environ.setdefault("seed", "42")  # celeba task.py reads int(os.environ["seed"]) at load

from celebanoniid.celebanoniid.task import load_data, _partition_cache_path

def main():
    K = int(sys.argv[1])
    t0 = time.time()
    for p in range(K):
        cp = _partition_cache_path(p, K)
        if cp.exists():
            print(f"[prewarm] K={K} p={p}: cached", flush=True)
            continue
        t = time.time()
        # return_sensitive=True builds the full cache (train+test+sensitive) so
        # every downstream caller (fair metrics included) hits the fast path.
        load_data(p, K, return_sensitive=True)
        print(f"[prewarm] K={K} p={p}: built in {time.time()-t:.0f}s", flush=True)
    print(f"[prewarm] K={K} COMPLETE ({K} partitions, {time.time()-t0:.0f}s total)", flush=True)

if __name__ == "__main__":
    main()
