# template-backend-python

House template for a Python/FastAPI backend — based on [tiangolo/full-stack-fastapi-template](https://github.com/tiangolo/full-stack-fastapi-template)'s backend (auth, SQLModel + Alembic migrations, email, Sentry), with the frontend half removed (the house frontend is the separate unified Expo/React Native Web template) and deployment retargeted to AWS Lambda via Terraform. Recommended when scaffolding an **ML/AI-heavy** backend (see the `new-app` skill).

## Requirements

- [Docker](https://www.docker.com/)
- [uv](https://docs.astral.sh/uv/) for Python package/environment management

## Local development

```bash
cp .env.example .env        # then edit the "changethis" placeholders
docker compose up -d db mailpit
uv sync
uv run bash scripts/prestart.sh   # runs migrations
uv run fastapi dev app/main.py
```

API at `http://localhost:8000`, interactive docs at `http://localhost:8000/docs`, Mailpit (dev email capture) at `http://localhost:8025`.

Or run everything in Docker: `docker compose up`.

## Testing & linting

```bash
uv run bash scripts/tests-start.sh   # pytest, with coverage
uv run bash scripts/lint.sh          # ruff + mypy
uv run bash scripts/format.sh
```

## Deploying

Defaults to **AWS Lambda** (via `Dockerfile.lambda` + Mangum — see `app/lambda_handler.py`), not the plain `Dockerfile` (that one's for local/compose use only). See [`infra/README.md`](infra/README.md) for one-time AWS account setup and per-app `terraform apply`. CI (`.github/workflows/deploy.yml`) builds/pushes the Lambda image and updates the function on every push to `main`.

## Working in this repo with Claude Code

See [`.claude/CLAUDE.md`](.claude/CLAUDE.md) and [`specs/README.md`](specs/README.md) for the spec-driven-development convention this repo follows.
