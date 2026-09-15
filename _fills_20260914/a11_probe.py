import tensorflow as tf
from celebanoniid.celebanoniid.task import load_data
from attack_metric import calculate_pgd_score
_,_,x,y,_,_ = load_data(0,1,return_sensitive=True,sensitive_attr="Male",test_only=True)
m = tf.keras.models.load_model("../420/celebanoniid/4/42/global_model_round_10.keras", compile=False)
for lbl, (e,a,i) in {
    "in-loop proxy (0.03, 0.007, 7)": (0.03,0.007,7),
    "eps 0.03 at eval depth (40)":    (0.03,0.007,40),
    "post-hoc eval (0.01,0.0025,40)": (0.01,0.0025,40),
}.items():
    print(f"{lbl:34s} res = {calculate_pgd_score(m,x,'celebanoniid',y,epsilon=e,alpha=a,num_iter=i):.4f}")
