# infra/ — Terraform → AWS (Lambda)

Deploys as a **container-image Lambda function** behind a Function URL with response streaming enabled — this covers both plain CRUD endpoints and streaming AI-agent/chat endpoints (2026's Lambda response-streaming support removed the old reason to default to a container platform for that). If this app instead needs a steady, always-on process or very long-lived (30+ min) streams, switch to the Fargate pattern in `template-backend-java`'s `infra/` instead.

## One-time account setup (do this once, not per app)

Every app scaffolded from any of the house templates shares **one** Terraform state backend and **one** GitHub OIDC provider.

1. **Remote state backend** (S3 bucket + DynamoDB lock table):
   ```bash
   aws s3api create-bucket --bucket <your-house-terraform-state> --region us-east-1
   aws s3api put-bucket-versioning --bucket <your-house-terraform-state> --versioning-configuration Status=Enabled
   aws dynamodb create-table --table-name <your-house-terraform-locks> \
     --attribute-definitions AttributeName=LockID,AttributeType=S \
     --key-schema AttributeName=LockID,KeyType=HASH \
     --billing-mode PAY_PER_REQUEST
   ```
2. **GitHub OIDC provider**:
   ```bash
   aws iam create-open-id-connect-provider \
     --url https://token.actions.githubusercontent.com \
     --client-id-list sts.amazonaws.com \
     --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
   ```
   Save the ARN — every app's `terraform.tfvars` needs it.

Skip this section entirely if you already did it for another app built from these templates — reuse the same bucket/table/provider.

## Per-app setup

1. Copy `infra/backend.hcl.example` → `infra/backend.hcl` (gitignored) and fill in the shared bucket/table.
2. Copy `infra/environments/dev.tfvars.example` → `infra/environments/dev.tfvars` and fill in `github_oidc_provider_arn` + `github_repo`.
3. Build and push the **first** image manually (Terraform can't create a Lambda function pointing at a tag that doesn't exist yet in ECR):
   ```bash
   cd infra && terraform init -backend-config=backend.hcl
   terraform apply -target=aws_ecr_repository.app -var-file=environments/dev.tfvars
   cd ..
   aws ecr get-login-password | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
   docker build -f Dockerfile.lambda -t <ecr-repo-url>:latest .
   docker push <ecr-repo-url>:latest
   ```
4. Then the full apply: `terraform apply -var-file=environments/dev.tfvars`.
5. Put the `deploy_role_arn` output into the repo's GitHub Actions repo variables (e.g. `AWS_DEPLOY_ROLE_ARN`) so CI can assume it — see `.github/workflows/deploy.yml`.

## Before this is production-safe

- The Function URL is public (`authorization_type = "NONE"`) with `allow_origins = ["*"]` — tighten CORS to the real frontend origin(s) before shipping, and consider `AWS_IAM` auth + a signing proxy if the endpoint shouldn't be fully public.
- No custom domain wired up — Lambda Function URLs get an AWS-generated HTTPS URL by default; add API Gateway + a custom domain later if needed.
- The Lambda execution role starts with only CloudWatch Logs permissions. Add further least-privilege statements only as the app actually needs them (e.g. a named Secrets Manager ARN) — never a wildcard.
- Have `infra-reviewer` look at any Terraform diff before `terraform apply` in a real environment, not just CI's automated plan.
