variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "app_name" {
  type    = string
  default = "template-backend-python"
}

variable "environment" {
  type        = string
  description = "dev | staging | prod"
  default     = "dev"
}

variable "image_tag" {
  type        = string
  description = "Lambda container image tag — CI sets this to the git SHA on every push to main."
  default     = "latest"
}

variable "memory_size" {
  type    = number
  default = 512
}

variable "timeout" {
  type        = number
  description = "Seconds. Response streaming raises the practical ceiling to 15 min if a request needs it."
  default     = 30
}

variable "github_oidc_provider_arn" {
  type        = string
  description = "ARN of the account's GitHub OIDC provider (created once by the house bootstrap — see infra/README.md)."
}

variable "github_repo" {
  type        = string
  description = "\"owner/repo\" allowed to assume the deploy role via OIDC."
}
