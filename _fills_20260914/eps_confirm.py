import os, tensorflow as tf
from adult.adult.task import load_data
from attack_metric import calculate_pgd_score
_,_,x_test,y_test,_,_ = load_data(0,1,return_sensitive=True,sensitive_attr="sex")
for seed, rnd, shipped in [("42","5",0.890872), ("123","10",0.915212), ("107","2",0.891408)]:
    m = tf.keras.models.load_model(f"../420/adult/20/{seed}/global_model_round_{rnd}.keras", compile=False)
    s = calculate_pgd_score(m, x_test, "adult", y_test, epsilon=0.03, alpha=0.007, num_iter=40)
    print(f"seed {seed:>4} round {rnd:>2}: eps=0.03 -> {s:.6f}   shipped {shipped:.6f}   diff {abs(s-shipped):.2e}")
