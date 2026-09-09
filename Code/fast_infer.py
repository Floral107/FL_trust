"""fast_infer — shared compiled forward-pass helper for in-loop scoring.

The in-loop GTG/Shapley scoring calls `model(x, training=False)` hundreds of
times per round (clean preds, PGD steps, noise-sampling). In pure eager mode on
CPU this dominates the wall-clock (celeba K=20 DY cells: ~7h per round). Wrapping
the forward pass in a `tf.function` that is **compiled once and reused** across
all those calls is a result-preserving speed-up (identical math, just a fused
graph instead of per-op eager dispatch).

Two subtleties handled here:
  * The compiled fn is cached per *model object* (WeakKeyDictionary) so it traces
    once and is reused across the many scoring calls. set_weights() only mutates
    the existing variables' values; it does NOT invalidate the traced graph.
  * The batch size varies per call (the clean-correct subset differs per
    coalition). `reduce_retracing=True` makes tf.function generalise over the
    batch dimension after a couple of traces, so it does not retrace endlessly.
"""
import weakref
import tensorflow as tf

_INFER_CACHE = weakref.WeakKeyDictionary()  # model -> compiled forward fn


def get_infer(model):
    """Return a cached `@tf.function` that computes model(x, training=False)."""
    fn = _INFER_CACHE.get(model)
    if fn is None:
        @tf.function(reduce_retracing=True)
        def _forward(x):
            return model(x, training=False)
        try:
            _INFER_CACHE[model] = fn = _forward
        except TypeError:
            fn = _forward  # model not weak-referenceable; still works, no caching
    return fn
