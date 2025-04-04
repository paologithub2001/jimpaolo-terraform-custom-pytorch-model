# Defines an IAM policy document that allows AWS SageMaker to assume a role.
data "aws_iam_policy_document" "sagemaker_assume_role_policy" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

# This creates an IAM role named sagemaker_access_iam_role
# The role uses the assume role policy defined earlier
resource "aws_iam_role" "sagemaker_access_iam_role" {
  name               = "sagemaker_access_iam_role"
  path               = "/system/"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume_role_policy.json
}

# Attaches the AmazonSageMakerFullAccess policy to the role.
# This grants SageMaker the necessary permissions to create models, endpoints, and perform training.
resource "aws_iam_role_policy_attachment" "sagemaker_access_policy_attachment" {
  role       = aws_iam_role.sagemaker_access_iam_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# Variable for the container image that will be used for inference.
variable container_image {}

# Creates an SageMaker model named mnist-model
# The model will run inside a Docker container specified by var.container_image
resource "aws_sagemaker_model" "pretrained-vit-model" {
  name               = "pretrained-vit-model"
  execution_role_arn = aws_iam_role.sagemaker_access_iam_role.arn

  primary_container {
    image = var.container_image
  }
}

provider "aws" {
  region = "ap-southeast-2"  # Set this to your AWS region
  profile = "default"        # If using named profiles from AWS CLI
}

# Endpoint configuration
resource "aws_sagemaker_endpoint_configuration" "pretrained-vit-configuration" {
  name = "my-endpoint-config"

  production_variants {
    variant_name           = "pretrained-vit-variant"
    model_name             = aws_sagemaker_model.pretrained-vit-model.name
    initial_instance_count = 1
    instance_type          = "ml.g4dn.xlarge"
  }

  tags = {
    Name = "pretrained-vit-config"
  }
}

# Endpoint
# This endpoint serves the model and allows it to be used for real-time inference
resource "aws_sagemaker_endpoint" "pretrain_vit_endpoint" {
  name                 = "pretrained-vit-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.pretrained-vit-configuration.name

  tags = {
    Name = "pretrained-vit-endpoint"
  }
}

output "sagemaker_endpoint_name" {
    value       = aws_sagemaker_endpoint.pretrain_vit_endpoint.name
    description = "SageMaker Endpoint Name"
}

output "sagemaker_endpoint_arn" {
    value       = aws_sagemaker_endpoint.pretrain_vit_endpoint.arn
    description = "SamgeMake Endpoint ARN"
}

