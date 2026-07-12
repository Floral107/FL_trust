"""RobShap demo — a clickable dashboard over the FLRobustness contribution-scoring pipeline.

Two modes:
  * Explore  — browse the paper's precomputed per-client trustworthiness contributions.
  * Live     — upload a tabular dataset split into clients and score contributions on the fly.

Nothing in here touches the research code; it only reads its CSV outputs and re-implements a
lightweight (CPU, seconds) analogue of the Leave-One-Out contribution scoring for live uploads.
"""

__version__ = "0.1.0"
