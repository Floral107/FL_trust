import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gc
import numpy as np
import tensorflow as tf

from robustness_metric import load_client_models, calculate_robustness_score, aggregate_models
from attack_metric import calculate_pgd_score, calculate_cw_score
from fairness_metric import calculate_fairness_score
def _process_text(x):
    # Lazy import: hosts that only deploy *noniid app dirs have no imdb/ package.
    from imdb.imdb.task import process_text
    return process_text(x)
from gtg_shap import GTGShapley


from privacy_metric import (
    make_mia_splits,
    mia_privacy_score_for_coalition,
    MIAConfig,
    MIASplits,
    prepare_mia_inputs,
)


def _build_privacy_value(method, x_test_priv, y_test, dataset, round_num, mia_seed, mia_splits, mia_config, reference_model=None):
    """Returns a privacy scoring callable, or None if method is not privacy-related.

    For GTG, pass reference_model so we can create a reusable ft_template
    (avoids clone_model overhead per coalition subset).
    """
    if method not in ("l1o_priv", "gtg_priv"):
        return None

    if mia_splits is None:
        mia_splits = make_mia_splits(
            y_test, seed=mia_seed, round_num=round_num,
            member_ratio=0.5, attack_train_ratio=0.5, balance=True,
        )
    if mia_config is None:
        mia_config = MIAConfig(
            clip_values=(0.0, 1.0) if "cifar" in dataset.lower() else None,
        )

    prepared = prepare_mia_inputs(x_test_priv, y_test, mia_splits)

    # For GTG: create a single ft_template to reuse across all coalition evals.
    # Compile it once here so _finetune_on_cpu only resets optimizer state.
    ft_template = None
    if method == "gtg_priv" and reference_model is not None:
        units = reference_model.output_shape[-1] if len(reference_model.output_shape) >= 2 else 1
        if units == 1:
            ft_loss = tf.keras.losses.BinaryCrossentropy(from_logits=mia_config.use_logits)
        else:
            ft_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=mia_config.use_logits)
        with tf.device("/CPU:0"):
            ft_template = tf.keras.models.clone_model(reference_model)
            ft_template.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=mia_config.finetune_lr),
                loss=ft_loss,
                metrics=["accuracy"],
            )

    def _privacy_value(mdl) -> float:
        out = mia_privacy_score_for_coalition(
            mdl, x_test_priv, y_test,
            seed=mia_seed, round_num=round_num,
            splits=mia_splits, config=mia_config,
            prepared=prepared,
            ft_template=ft_template,
        )
        print(f"[MIA] round={round_num} privacy={out['privacy']:.4f} auc={out['mia_auc']:.4f} acc={out['mia_acc']:.4f}")
        return float(out["privacy"])

    return _privacy_value


def _accuracy_score(model, x_test, y_test, dataset):
    """Plain top-1 accuracy. Manual argmax/threshold avoids relying on
    `model.compile(..., metrics=['accuracy'])` which the clone_model paths
    in l1o_evals / gtg_evals don't reliably set up."""
    x_eval = _process_text(x_test) if "imdb" in dataset.lower() else x_test
    preds = model.predict(x_eval, verbose=0, batch_size=512)
    if preds.ndim >= 2 and preds.shape[-1] == 1:
        pred_labels = (preds.flatten() > 0.5).astype(np.int32)
    elif preds.ndim >= 2:
        pred_labels = np.argmax(preds, axis=-1)
    else:
        pred_labels = (preds > 0.5).astype(np.int32)
    y = y_test.numpy() if hasattr(y_test, "numpy") else np.asarray(y_test)
    return float(np.mean(pred_labels == y))


def _loss_score(model, x_test, y_test, dataset):
    """Per-sample mean cross-entropy. Uses BCE for adult (binary sigmoid head),
    SparseCategoricalCrossentropy elsewhere — matches the training-time loss.
    Computed manually so we don't depend on the model being recompiled with
    the right loss object after clone_model."""
    x_eval = _process_text(x_test) if "imdb" in dataset.lower() else x_test
    preds = model.predict(x_eval, verbose=0, batch_size=512)
    y = y_test.numpy() if hasattr(y_test, "numpy") else np.asarray(y_test)
    if "adult" in dataset.lower():
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        y_in = y.astype(np.float32)
        p_in = preds.flatten() if (preds.ndim >= 2 and preds.shape[-1] == 1) else preds
        return float(loss_fn(y_in, p_in).numpy())
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    return float(loss_fn(y, preds).numpy())


def _compute_global_metric(method, global_model, x_test, y_test, dataset, fairness_wrapper, privacy_value):
    if method in ("l1o_rob", "gtg_rob"):
        return float(calculate_robustness_score(global_model, x_test, dataset))
    # adv / adv_pgd: kept together — l1o_adv / gtg_adv are backward-compat PGD aliases
    if method in ("l1o_adv", "gtg_adv", "l1o_adv_pgd", "gtg_adv_pgd"):
        return float(calculate_pgd_score(global_model, x_test, dataset, y_test))
    if method in ("l1o_adv_cw", "gtg_adv_cw"):
        return float(calculate_cw_score(global_model, x_test, dataset, y_test))
    if method in ("l1o_fair", "gtg_fair"):
        return float(fairness_wrapper(global_model, x_test, "dp"))
    if method in ("l1o_fair_eo", "gtg_fair_eo"):
        return float(fairness_wrapper(global_model, x_test, "eo"))
    if method in ("l1o_priv", "gtg_priv"):
        return float(privacy_value(global_model))
    if method in ("l1o_acc", "gtg_acc"):
        return _accuracy_score(global_model, x_test, y_test, dataset)
    if method in ("l1o_loss", "gtg_loss"):
        return _loss_score(global_model, x_test, y_test, dataset)
    return None


def evaluate_round(
    model_dir,
    num_clients,
    x_test,
    y_test,
    s_test,
    round_num,
    dataset,
    method,
    *,
    mia_splits: MIASplits | None = None,
    mia_config: MIAConfig | None = None,
    mia_seed: int = 42,
):
    """
    Returns:
        (global_metric, contributions_dict)

    Privacy (l1o_priv / gtg_priv):
      - coalition-model is cloned and fine-tuned on a fixed member split of TEST
      - combined features (loss, confidence, entropy, correctness) fed to RF attack classifier
      - privacy score is higher=better, in [0,1]
    """
    global_model_path = os.path.join(model_dir, f"global_model_round_{round_num}.keras")
    if not os.path.exists(global_model_path):
        print(f"Global model for round {round_num} not found: {global_model_path}")
        return None, None

    try:
        global_model = tf.keras.models.load_model(global_model_path)
    except Exception as e:
        print(f"Failed to load global model: {e}")
        return None, None

    client_models = load_client_models(model_dir, num_clients, round_num)
    if not client_models:
        print(f"No client models found in {model_dir} for round {round_num}")
        return None, None

    # Extract weights immediately, then free the Keras model objects.
    # For GTG/privacy we only need numpy weight arrays, not full models.
    client_weights = {cid: model.get_weights() for cid, model in client_models.items()}
    client_id_list = sorted(client_weights.keys())
    index_to_client_id = dict(enumerate(client_id_list))

    # Free client model graphs — weights are kept as numpy arrays
    del client_models
    gc.collect()

    # Embed imdb text ONCE per round. GTG/LOO call the metric on hundreds of
    # coalitions; without this each call re-ran USE over all 5000 test texts
    # (~5s/eval → ~50min/round). process_text is idempotent on already-embedded
    # (N,512) input, so every downstream metric/PGD call is now a no-op pass-through.
    if "imdb" in dataset.lower():
        x_test = _process_text(x_test)

    def fairness_wrapper(model, X, fairness_metric: str):
        return calculate_fairness_score(model, X, s_test, dataset, y_test=y_test, fairness_metric=fairness_metric)

    x_test_priv = _process_text(x_test) if "imdb" in dataset.lower() else x_test
    privacy_value = _build_privacy_value(method, x_test_priv, y_test, dataset, round_num, mia_seed, mia_splits, mia_config, reference_model=global_model)

    try:
        global_metric = _compute_global_metric(method, global_model, x_test, y_test, dataset, fairness_wrapper, privacy_value)
        if global_metric is None:
            print(f"Unsupported method: {method}")
            return None, None
    except Exception as e:
        print(f"Error computing global metric ({method}): {e}")
        return None, None

    try:
        metric_fun = _get_metric_fun(method, fairness_wrapper, privacy_value)
        x_eval = x_test_priv if method in ("l1o_priv", "gtg_priv") else x_test
        if method.startswith("l1o_"):
            global_val, contribs = l1o_evals(metric_fun, global_metric, x_eval, y_test, dataset, client_weights, global_model)
        elif method.startswith("gtg_"):
            # loss is a lower-is-better metric: GTG-Shapley best-subset selection
            # must minimize (loss_flag=True), else it keeps the worst clients and
            # zeros the good ones. Env toggle FLR_GTG_LOSSFLAG=0 reproduces the
            # old (incorrect) behavior for A/B comparison.
            gtg_loss_flag = method.endswith("_loss") and os.environ.get("FLR_GTG_LOSSFLAG", "1") != "0"
            global_val, contribs = gtg_evals(metric_fun, model_dir, num_clients, x_eval, y_test, round_num, dataset, client_weights, index_to_client_id, global_metric, loss_flag=gtg_loss_flag)
        else:
            return None, None
        # CANONICAL SIGN CONVENTION: contribution values are higher=better for
        # EVERY metric. Loss utilities are evaluated raw (lower=better), so the
        # marginal contributions come out inverted — negate them HERE, the single
        # point the driver (robustness.py — the ONLY post-hoc driver; the stale
        # loss.py/accuracy.py with their own flips live in _archive/) routes
        # through. global_* stays raw CE. Sign logic must exist nowhere else
        # (no flips in merge/collect/plot scripts — that caused the 2026-06/07
        # double-flip mess); sanity_gate.py enforces the convention on data/combo.
        if method.endswith("_loss") and contribs is not None:
            contribs = {k: -v for k, v in contribs.items()}
        return global_val, contribs
    finally:
        del global_model
        gc.collect()
        tf.keras.backend.clear_session()
        gc.collect()


def _get_metric_fun(method: str, fairness_wrapper, privacy_value):
    """
    Returns f(model, X, dataset, y_test) -> float
    compatible with l1o_evals / gtg_evals
    """
    if method.endswith("_rob"):
        return lambda model, X, dataset, y_test: float(calculate_robustness_score(model, X, dataset))
    # CW is tested before the generic _adv suffix so *_adv_cw doesn't fall into the PGD branch.
    if method.endswith("_adv_cw"):
        return lambda model, X, dataset, y_test: float(calculate_cw_score(model, X, dataset, y_test))
    if method.endswith("_adv") or method.endswith("_adv_pgd"):
        return lambda model, X, dataset, y_test: float(calculate_pgd_score(model, X, dataset, y_test))
    if method.endswith("_fair_eo"):
        return lambda model, X, dataset, y_test: float(fairness_wrapper(model, X, "eo"))
    if method.endswith("_fair"):
        return lambda model, X, dataset, y_test: float(fairness_wrapper(model, X, "dp"))
    if method.endswith("_priv"):
        if privacy_value is None:
            raise ValueError("privacy_value is None for *_priv.")
        return lambda model, X, dataset, y_test: float(privacy_value(model))
    if method.endswith("_acc"):
        return lambda model, X, dataset, y_test: _accuracy_score(model, X, y_test, dataset)
    if method.endswith("_loss"):
        return lambda model, X, dataset, y_test: _loss_score(model, X, y_test, dataset)
    raise ValueError(f"Unsupported method: {method}")


def l1o_evals(metric_fun, global_metric, x_test, y_test, dataset, client_weights, global_model):
    contributions = {}

    # keep old compile behavior for safety (PGD / etc.)
    loss_fn_name = "binary_crossentropy" if "adult" in dataset.lower() else "sparse_categorical_crossentropy"

    # Clone-once: building a fresh model + compiling it is the dominant cost when
    # the per-coalition metric is cheap (fair / acc / loss). On CPU this saves
    # ~15-25% per round and scales with num_clients. We rebuild weights per
    # coalition instead of cloning the whole graph.
    temp_model = tf.keras.models.clone_model(global_model)
    temp_model.compile(optimizer="adam", loss=loss_fn_name)

    try:
        for client_id in client_weights.keys():
            remaining_weights = [client_weights[cid] for cid in client_weights if cid != client_id]
            if not remaining_weights:
                continue

            try:
                temp_model.set_weights(aggregate_models(remaining_weights))
                without_i = float(metric_fun(temp_model, x_test, dataset, y_test))
                contributions[client_id] = float(global_metric - without_i)  # V(All) - V(All\i)
            except Exception as e:
                print(f"Error in client {client_id} LOO calc ({dataset}): {e}")
    finally:
        try:
            del temp_model
        except NameError:
            pass
        gc.collect()

    return float(global_metric), contributions


def gtg_evals(metric_fun, model_dir, num_clients, x_test, y_test, round_num,
              dataset, client_weights, index_to_client_id, global_metric, loss_flag=False):

    rounds_to_load = [r for r in {round_num - 1, round_num} if r >= 0]
    loaded_models = load_global_models(model_dir, rounds_to_load, dataset)

    prev_model = loaded_models.get(round_num - 1, loaded_models[round_num])
    eval_model = tf.keras.models.clone_model(loaded_models[round_num])

    def utility_function(subset):
        try:
            if len(subset) == 0:
                return float(metric_fun(prev_model, x_test, dataset, y_test))

            selected_weights = [client_weights[index_to_client_id[i]] for i in subset]
            eval_model.set_weights(aggregate_models(selected_weights))
            return float(metric_fun(eval_model, x_test, dataset, y_test))

        except Exception as e:
            print(f"Error in subset {subset} utility ({dataset}): {e}")
            return 0.0

    gtg = GTGShapley(num_players=num_clients, loss_flag=loss_flag)
    gtg.set_utility_function(utility_function)

    try:
        vals = gtg.compute(round_num=round_num)
        contributions = {index_to_client_id[i]: float(vals[i]) for i in range(gtg.num_players)}
        return float(global_metric), contributions

    finally:
        try:
            gtg.evaluated_subsets.clear()
            gtg.shapley_values.clear()
            gtg.shapley_values_best_subset.clear()
        except Exception:
            pass

        try:
            del eval_model
        except Exception:
            pass

        try:
            for m in loaded_models.values():
                del m
        except Exception:
            pass

        del loaded_models, gtg
        gc.collect()


def load_global_models(model_dir, rounds, dataset):
    loaded_models = {}
    loss_fn = "binary_crossentropy" if "adult" in dataset.lower() else "sparse_categorical_crossentropy"

    for r in rounds:
        model_path = os.path.join(model_dir, f"global_model_round_{r}.keras")
        if not os.path.exists(model_path):
            continue
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer="adam", loss=loss_fn)
        loaded_models[r] = model

    if not loaded_models:
        raise ValueError(f"No global models found under {model_dir} for rounds={rounds}")

    return loaded_models