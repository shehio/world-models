terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# -----------------------------------------------------------------------------
# AMI lookup
#
# Default: latest "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)"
# which has CUDA drivers preinstalled. Works on CPU instances too (it's just an
# Ubuntu image with some extra packages; it doesn't refuse to boot without a GPU).
# Override via var.ami_id if you need a specific build.
# -----------------------------------------------------------------------------
data "aws_ami" "default" {
  count       = var.ami_id == "" ? 1 : 0
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

locals {
  ami_id = var.ami_id != "" ? var.ami_id : data.aws_ami.default[0].id

  common_tags = merge(
    {
      Project   = var.project_tag
      ManagedBy = "terraform"
    },
    var.extra_tags,
  )
}

# -----------------------------------------------------------------------------
# Networking — use the default VPC. This is a one-off experiment box, no need
# to build a VPC from scratch.
# -----------------------------------------------------------------------------
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "aws_security_group" "experiment" {
  name        = "${var.project_tag}-sg"
  description = "SSH-only access to the 02c experiment box"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH from allowed CIDR"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  egress {
    description = "All egress (apt, pip, github, etc.)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

# -----------------------------------------------------------------------------
# Bootstrap script — installed via user_data on first boot. Idempotent.
# Installs uv (modern Python package manager), stockfish, and prepares a workspace.
# -----------------------------------------------------------------------------
locals {
  user_data = templatefile("${path.module}/user_data.sh", {
    project_tag = var.project_tag
  })
}

# -----------------------------------------------------------------------------
# The instance.
#
# Spot vs on-demand is controlled by var.use_spot. AWS's modern way to do this
# from inside aws_instance is the instance_market_options block.
# -----------------------------------------------------------------------------
resource "aws_instance" "experiment" {
  ami                         = local.ami_id
  instance_type               = var.instance_type
  key_name                    = var.key_name
  vpc_security_group_ids      = [aws_security_group.experiment.id]
  subnet_id                   = data.aws_subnets.default.ids[0]
  associate_public_ip_address = true
  user_data                   = local.user_data

  root_block_device {
    volume_size           = var.volume_size_gb
    volume_type           = var.volume_type
    delete_on_termination = true
    tags                  = local.common_tags
  }

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        instance_interruption_behavior = "terminate"
        spot_instance_type             = "one-time"
        max_price                      = var.spot_max_price == "" ? null : var.spot_max_price
      }
    }
  }

  tags = merge(local.common_tags, { Name = "${var.project_tag}-${var.instance_type}" })

  # Ensure root EBS is large enough before we install heavy CUDA stuff via user_data.
  lifecycle {
    precondition {
      condition     = var.volume_size_gb >= 100
      error_message = "volume_size_gb must be >= 100 (DLAMI itself uses ~60GB)."
    }
  }
}
