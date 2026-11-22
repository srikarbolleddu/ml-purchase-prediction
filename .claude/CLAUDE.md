# template-backend-python

FastAPI backend (Python 3.13), based on tiangolo/full-stack-fastapi-template's backend half. Deploys to AWS Lambda via Terraform (`infra/`) — see `infra/README.md` before the first deploy. Recommended template for ML/AI-heavy apps.

## Stack

- FastAPI + SQLModel + Alembic migrations, Postgres, JWT auth already wired
- Package/env management: `uv` (not pip/poetry directly)
- Local run: `docker compose up -d db mailpit && uv run fastapi dev app/main.py`
- Tests: `uv run bash scripts/tests-start.sh` (pytest + coverage, threshold 90%)
- Lint/format: `uv run bash scripts/lint.sh`, `uv run bash scripts/format.sh` (ruff + mypy)
- Health check: `GET /health`

## Spec-driven development

Real features go through `/feature-dev`. Save its Discovery output to `specs/<feature-slug>/spec.md` and its chosen Architecture to `specs/<feature-slug>/plan.md` before implementation starts — see `specs/README.md`. Skip this ceremony for trivial, single-file changes.

## Conventions

- Routes in `app/api/`, DB models in `app/models.py` (SQLModel), business logic doesn't belong in route handlers for anything non-trivial.
- New AWS permissions go on `aws_iam_role.lambda_exec` in `infra/main.tf` as a named, least-privilege statement — never a wildcard.
- Claude API calls go through `app/core/claude_client.py` (server-side only — never expose the API key to a client). See that file for the pattern.
- Don't hand-edit `infra/` Terraform freehand for anything beyond small variable tweaks — for new infrastructure, use the `terraform-aws-deploy` skill so it stays consistent with the other apps built from these templates.
