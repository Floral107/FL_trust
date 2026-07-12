"""eps_calibrate.py — pick the IMDB PGD eval epsilon on a CONVERGED pilot model.

Advisor comment #5: at alpha=eps=0.0015 the attack barely perturbs anything, so
res ~= clean acc and ET-adv can never beat BL. We need an eps where the BL model
is meaningfully degraded (res in [0.3, 0.6]) so adversarial training has room to
be a credible ceiling.

Runs on the pilot BL model (imdbnoniid K20 seed42 round 10, new LR). Writes the
chosen eps to imdb_eps.txt. The driver then WAITS until attack_metric.py /
score_metrics.py are updated with this eps and pushed (sentinel EPS_APPLIED) —
a deliberate human checkpoint: eyeball the curve before committing the fleet.

Run: CUDA_VISIBLE_DEVICES= seed=42 python3 eps_calibrate.py
"""
import os
os.environ.setdefault("seed", "42")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import tensorflow as tf
from attack_metric import calculate_pgd_text_robustness
from imdbnoniid.imdbnoniid.task import load_data

CANDIDATES = [0.0015, 0.005, 0.01, 0.02, 0.05, 0.1]
STEPS = 10          # stronger than the old 5; matches ADULT/CelebA's regime
TARGET = (0.3, 0.6)
MODEL = "420/imdbnoniid/20/42/global_model_round_10.keras"

def main():
    _, _, x_test, y_test = load_data(0, 1)
    model = tf.keras.models.load_model(MODEL)
    results = []
    for eps in CANDIDATES:
        res = calculate_pgd_text_robustness(
            model, x_test, y_test, epsilon=eps, alpha=eps / 4, num_iter=STEPS)
        results.append((eps, res))
        print(f"[calib] eps={eps:.4f} alpha={eps/4:.5f} steps={STEPS} -> BL res={res:.3f}", flush=True)
    chosen = next((e for e, r in results if TARGET[0] <= r <= TARGET[1]), None)
    with open("imdb_eps_curve.txt", "w") as f:
        for e, r in results:
            f.write(f"{e}\t{r}\n")
    if chosen is None:
        print("[calib] NO eps hit the target band — curve in imdb_eps_curve.txt; manual pick needed")
        return 1
    with open("imdb_eps.txt", "w") as f:
        f.write(f"{chosen}\n")
    print(f"[calib] chosen eps={chosen} (alpha=eps/4, steps={STEPS}) -> imdb_eps.txt")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
