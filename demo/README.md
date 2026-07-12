# RobShap demo

A clickable dashboard for the **FLRobustness** thesis pipeline: per-client contribution
scoring in federated learning across **robustness / adversarial / fairness / privacy** —
turning "which client helps, which hurts, and at what cost?" into something you can
explore in a browser.

![CI](https://img.shields.io/badge/CI-lint%20%2B%20pytest%20%2B%20docker-blue)

## What it does

| Tab | What happens |
|---|---|
| **Explore paper results** | Browse the thesis' precomputed per-client L1O & GTG-Shapley contributions (CIFAR-10 / IMDB / Adult / CelebA, IID & non-IID, 4 & 20 clients, 5 seeds, 10 rounds). Trade-off scatter, per-client radar, round timelines. |
| **Score your own clients** | Upload a CSV with a `client` column and a target column. The app runs real Leave-One-Out contribution scoring with a fast sklearn surrogate — accuracy, loss, noise robustness, and DP/EO fairness — in seconds on CPU. A bundled sample includes a label-flipped client and a group-biased client so the scores tell a story out of the box. |

**Scope note:** the paper's expensive axes (certified robustness via randomized smoothing,
PGD adversarial, MIA privacy) need GPU-hours per coalition and are served from precomputed
research runs; live uploads score the cheap-but-faithful axes only.

## Run it

```bash
# Local
pip install -r requirements.txt
python make_sample_data.py
streamlit run app.py

# Docker
docker build -t robshap-demo .
docker run -p 8501:8501 robshap-demo
```

Open http://localhost:8501.

## Layout

```
demo/
├── app.py                  # Streamlit UI (3 tabs)
├── robshap_demo/
│   ├── loader.py           # normalizes research-pipeline CSVs → tidy frame
│   └── scoring.py          # live L1O scoring (sklearn surrogate, CPU)
├── data/                   # precomputed contribution CSVs (from the thesis runs)
├── sample_data/            # generated demo dataset (make_sample_data.py)
├── tests/                  # pytest smoke + behavior tests
└── Dockerfile
```

CI (`.github/workflows/demo-ci.yml`): ruff lint → pytest → docker build → container
health-check smoke test, on every push touching `demo/`.

## Deploy

Any container host works (the image is self-contained, no GPU):

- **Hugging Face Spaces** (free, easiest): push this folder as a Docker Space, port 8501.
- **Fly.io / Render / Railway**: point at the Dockerfile.

## Relation to the research code

The FL training (Flower + TensorFlow) and the full metric pipeline live in the repo root
and are untouched. This demo only **consumes** their CSV outputs and re-implements the
L1O scoring semantics on a lightweight model for interactive use.
