"""One reliability (rob) eval on a real CelebA model, alone on the box: wall time + peak memory.
Read-only: writes no results. Used to price the lane R jobs."""
import os, time, ctypes, ctypes.wintypes as wt

class PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

k32 = ctypes.windll.kernel32
k32.GetCurrentProcess.restype = wt.HANDLE
k32.K32GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]

def peak_gb():
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 1e9

import pandas as pd
import tensorflow as tf
from celebanoniid.celebanoniid.task import load_data
from robustness_metric import calculate_robustness_score

_, _, x, y, _, _ = load_data(0, 1, return_sensitive=True, sensitive_attr="Male", test_only=True)
m = tf.keras.models.load_model("../420/celebanoniid/4/42/global_model_round_10.keras", compile=False)
print(f"loaded: N={len(x)}  peak so far {peak_gb():.1f} GB", flush=True)
t = time.time()
r = calculate_robustness_score(m, x, "celebanoniid")
dt = time.time() - t
print(f"rel={r:.4f}  one eval (L=100, N={len(x)}, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}) = {dt:.0f} s  peak working set = {peak_gb():.1f} GB", flush=True)
c = pd.read_csv("data/combo/results_celebanoniid_l1o.csv")
c = c[(c.num_clients == 4) & (c["round"] == 10)]
print("shipped global_l1o_rob r10 (other seeds): " + str(c.groupby("seed")["global_l1o_rob"].first().round(4).to_dict()))
