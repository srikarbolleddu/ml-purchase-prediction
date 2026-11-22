locals {
  name = "${var.app_name}-${var.environment}"
}

# --- Container registry ---

resource "aws_ecr_repository" "app" {
  name                 = local.name
  image_tag_mutability = "IMMUTABLE"
  force_delete         = var.environment != "prod"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

# --- Logs ---

resource "aws_cloudwatch_log_group" "app" {
  name              = "/aws/lambda/${local.name}"
  retention_in_days = var.environment == "prod" ? 90 : 14
}

# --- Lambda execution role ---
# Task-specific AWS permissions go here as named, least-privilege
# statements — never a wildcard. Empty by design until the app needs one
# (e.g. Secrets Manager read for a specific secret ARN).

resource "aws_iam_role" "lambda_exec" {
  name = "${local.name}-lambda-exec"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "lambda_logs" {
  name = "${local.name}-logs"
  role = aws_iam_role.lambda_exec.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["logs:CreateLogStream", "logs:PutLogEvents"]
      Resource = "${aws_cloudwatch_log_group.app.arn}:*"
    }]
  })
}

# --- Lambda function (container image), built from Dockerfile.lambda ---

resource "aws_lambda_function" "app" {
  function_name = local.name
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
  memory_size   = var.memory_size
  timeout       = var.timeout

  environment {
    variables = {
      FASTAPI_ENV = var.environment == "dev" ? "development" : "production"
    }
  }

  depends_on = [aws_cloudwatch_log_group.app]
}

# Public HTTPS endpoint with response streaming — the simplest way to get a
# streaming-capable public URL for AI-agent/chat endpoints without standing
# up a separate API Gateway. CORS is handled here rather than in app code.
resource "aws_lambda_function_url" "app" {
  function_name      = aws_lambda_function.app.function_name
  authorization_type = "NONE"
  invoke_mode        = "RESPONSE_STREAM"

  cors {
    allow_origins = ["*"] # tighten to the real frontend origin(s) once known
    allow_methods = ["*"]
    allow_headers = ["*"]
  }
}
