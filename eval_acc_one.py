"""eval_acc_one.py — print held-out accuracy of one saved model. Used by the
beta-ablation to score each (mode,metric,seed,beta) cell cheaply (acc only).
Usage: python eval_acc_one.py <model_path.keras> <dataset>
"""
import sys, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, "/root/ulrich")
import numpy as np
import tensorflow as tf
from reweight_eval import load_test_data

path, ds = sys.argv[1], sys.argv[2]
if not os.path.exists(path):
    print("NA"); sys.exit(0)
d = load_test_data(ds)
x, y = d[0], d[1]
m = tf.keras.models.load_model(path, compile=False)
p = m.predict(x, verbose=0)
y = np.asarray(y)
if p.ndim > 1 and p.shape[1] > 1:          # softmax / multiclass
    pred = p.argmax(1)
    yt = y.argmax(1) if (y.ndim > 1 and y.shape[1] > 1) else y.ravel().astype(int)
else:                                       # sigmoid / binary
    pred = (p.ravel() > 0.5).astype(int)
    yt = y.ravel().astype(int)
print(round(float((pred == yt).mean()), 4))
