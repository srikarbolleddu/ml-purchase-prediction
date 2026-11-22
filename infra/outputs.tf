output "function_url" {
  value = aws_lambda_function_url.app.function_url
}

output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

output "lambda_function_name" {
  value = aws_lambda_function.app.function_name
}

output "deploy_role_arn" {
  description = "Put this in the repo's GitHub Actions repo variables for OIDC deploy"
  value       = aws_iam_role.deploy.arn
}
