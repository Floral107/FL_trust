"""Which PGD eps reproduces the shipped adult global_gtg_adv?"""
import os, numpy as np, tensorflow as tf
os.environ.setdefault("seed","9928")
from adult.adult.task import load_data
from attack_metric import calculate_pgd_score
_,_,x_test,y_test,_,_ = load_data(0, 1, return_sensitive=True, sensitive_attr="sex")
m = tf.keras.models.load_model("../420/adult/20/9928/global_model_round_1.keras", compile=False)
p = m.predict(x_test, verbose=0)
pred = (p.ravel()>0.5).astype(int) if p.shape[-1]==1 else p.argmax(1)
yt = y_test.ravel() if getattr(y_test,'ndim',1)==1 or y_test.shape[-1]==1 else y_test.argmax(1)
print(f"clean acc            = {(pred==np.asarray(yt).ravel()).mean():.6f}")
print(f"SHIPPED round-1 value= 0.801138")
for eps in (0.3, 0.1, 0.05, 0.03, 0.01, 0.007, 0.005):
    s = calculate_pgd_score(m, x_test, "adult", y_test, epsilon=eps, alpha=eps/4, num_iter=40)
    print(f"eps={eps:<6} alpha=eps/4  res = {s:.6f}")
