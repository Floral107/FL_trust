"""eval_resacc_one.py — print "<res> <acc>" for one saved model (res = PGD
resilience at the per-dataset eval eps). Used by the res-value check to decide
whether res-weighting at the chosen beta improves robustness or just costs acc.
Usage: python eval_resacc_one.py <model_path.keras> <dataset>
"""
import sys, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, "/root/ulrich")
import numpy as np
import tensorflow as tf
from reweight_eval import load_test_data
from attack_metric import calculate_pgd_score

path, ds = sys.argv[1], sys.argv[2]
if not os.path.exists(path):
    print("NA NA"); sys.exit(0)
d = load_test_data(ds)
x, y = d[0], d[1]
m = tf.keras.models.load_model(path, compile=False)
p = m.predict(x, verbose=0)
yy = np.asarray(y)
if p.ndim > 1 and p.shape[1] > 1:
    pred = p.argmax(1)
    yt = yy.argmax(1) if (yy.ndim > 1 and yy.shape[1] > 1) else yy.ravel().astype(int)
else:
    pred = (p.ravel() > 0.5).astype(int)
    yt = yy.ravel().astype(int)
acc = float((pred == yt).mean())
res = float(calculate_pgd_score(m, x, ds, y, epsilon=0.01, alpha=0.0025, num_iter=40))
print(f"{round(res,4)} {round(acc,4)}")
