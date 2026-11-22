"""AWS Lambda entry point. Wraps the same FastAPI `app` used locally/in ECS
with Mangum, so there's exactly one app to maintain regardless of where it
runs. Response streaming (for long-running/AI-agent endpoints) works via a
Lambda Function URL with `InvokeMode=RESPONSE_STREAM` — see infra/README.md.
"""

from mangum import Mangum

from app.main import app

handler = Mangum(app)
