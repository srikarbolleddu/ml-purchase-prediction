# Term-Deposit Purchase Prediction

Will a bank customer buy a term deposit if we call them? A **pruned decision tree** trained on
41,188 real marketing calls, served as a **live demo API** with an interactive UI. You can change a
customer's profile and see the probability, plus the exact tree rules that produced it.

- **Live demo:** standalone UI on GitHub Pages (`web/`), and embedded on
  [srikarbolleddu.com](https://srikarbolleddu.com/projects/purchase-prediction-system)
- **Stack:** scikit-learn · FastAPI · AWS Lambda (container) · API Gateway · React (Vite)

> Capstone project, UT Austin Post-Graduate Program in AI & Machine Learning. Rebuilt from
> scratch on an openly licensed dataset. The course's notebooks and data are not used or included.

## Results (held-out test set, 8,238 customers)

| | Value |
|---|---|
| Base purchase rate | 11.3% |
| Precision (of customers we'd call, share who buy) | **43.6%**, ~3.9× better than calling at random |
| Recall (share of all buyers we reach) | **58.1%** |
| ROC-AUC / PR-AUC | 0.792 / 0.452 |
| Final model | Pre-pruned tree, depth 10, 95 leaves, decision threshold 0.65 |

All numbers come from [`reports/metrics.json`](reports/metrics.json), which `training/train.py` writes.

### Why pruning mattered

| Tree | Leaves | Train PR-AUC | Test PR-AUC |
|---|---|---|---|
| Default (grown until pure) | 4,581 | 0.999 | 0.174 |
| Pre-pruned (depth / leaf-size limits) | 95 | 0.463 | **0.452** |
| Post-pruned (cost-complexity α) | 75 | 0.438 | 0.442 |

The unpruned tree memorizes the training set and generalizes worst. Both pruning strategies close
the train/test gap. See `reports/figures/pruning_curve.png`.

### What we deliberately left out
The dataset's `duration` column (call length) predicts the outcome almost perfectly, but it is only
known *after* the call ends. Using it would leak the answer into the model, so it's excluded, and
the API rejects it if sent.

### What drives purchases
The **economic context** of the call (employment levels, consumer confidence, Euribor rate)
dominates the tree's feature importance, ahead of anything about the customer. When a campaign runs
matters more than who is called. The demo lets you change both to see this for yourself.

## Using the API

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/schema` | Input fields with types, ranges, options and help text. The demo form is generated from this. |
| `GET /api/v1/samples` | Six real held-out customers (3 buyers, 3 non-buyers, including one mistake each way) |
| `GET /api/v1/explore` | Precomputed chart data: pruning curve, threshold sweep, feature importance, metrics |
| `POST /api/v1/predict` | Probability, yes/no at the tuned threshold, and the decision path |
| `GET /api/v1/docs` | Interactive OpenAPI docs |

## Run locally

```bash
uv sync
uv run python training/train.py         # downloads the pinned dataset (sha256-checked) and retrains
uv run fastapi dev app/main.py          # API on http://localhost:8000/api/v1/docs
```

Checks: `uv run ruff check . && uv run ty check && uv run coverage run -m pytest && uv run coverage report`.
The tests include **golden predictions**: the shipped model must reproduce the outputs recorded at
training time.

## How the notebook became a service

| Notebook step | Where it lives now |
|---|---|
| Load + clean data, drop leaky columns, encode target | `app/features.py` (`clean_raw`), unit-tested |
| Feature definitions (types, ranges, categories) | `app/features.py` (`FIELDS`), which also drives API validation and the demo form |
| Train, prune, compare, pick threshold | `training/train.py`: fixed seeds, 60/20/20 split, test set used once |
| Model file | `app/model_artifacts/model.skops`. skops loads only allow-listed types, unlike pickle. |
| "Interpretation" cells | `app/model.py`: decision-path rules returned with every prediction |

## Security and abuse limits (public demo)

- No secrets: the API needs none, and nothing visitors send is stored.
- Strict input validation (types, ranges, known categories, no extra fields) and a 4 KB body cap.
- Throttling: an API Gateway stage limit across all callers, plus a per-client rate limit in the app.
- Deploys use GitHub OIDC (no stored AWS keys). The deploy role trusts only this repo's
  approval-gated `production` environment.

## Data and license

- **Dataset:** [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing),
  Moro, S., Rita, P., & Cortez, P. (2014), licensed **CC BY 4.0**. Downloaded at training time, not
  redistributed here.
- **Code:** see [LICENSE](LICENSE).
