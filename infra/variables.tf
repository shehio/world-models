# All knobs for the AWS experiment box. The most important one is
# `instance_type` — change it to scale the experiment up or down.
#
# Recommended values for instance_type (running the 02c pipeline):
#   c7i.16xlarge   — CPU-only, cheapest fast option            (~$0.80/h spot)
#   g5.24xlarge    — 96 vCPU + 4x A10G, sweet spot             (~$2.50/h spot)  <-- default
#   g5.48xlarge    — 192 vCPU + 8x A10G, absolute fastest      (~$5/h spot)
#   p4d.24xlarge   — 8x A100, overkill for this workload       (~$10-15/h spot)

variable "instance_type" {
  description = "EC2 instance type. The whole point of this variable: swap this to scale compute up/down."
  type        = string
  default     = "g5.24xlarge"
}

variable "region" {
  description = "AWS region. Pick one with good spot availability for your chosen instance_type."
  type        = string
  default     = "us-east-1"
}

variable "key_name" {
  description = "Name of an existing AWS EC2 key pair in `region`. Created out-of-band via `aws ec2 create-key-pair` or the console."
  type        = string
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH in. Default opens to the world — TIGHTEN THIS to your IP for production."
  type        = string
  default     = "0.0.0.0/0"
}

variable "use_spot" {
  description = "If true, request a spot instance (cheaper but can be reclaimed). If false, on-demand."
  type        = bool
  default     = true
}

variable "spot_max_price" {
  description = "Max hourly price you'll pay for spot (USD). Leave empty for AWS's current spot price (recommended)."
  type        = string
  default     = ""
}

variable "volume_size_gb" {
  description = "Root EBS volume size in GB. 02c data + checkpoints fit easily in 100; bump for larger experiments."
  type        = number
  default     = 200
}

variable "volume_type" {
  description = "EBS volume type. gp3 is the right default — fast and cheap."
  type        = string
  default     = "gp3"
}

variable "ami_id" {
  description = "Override AMI. Leave empty to auto-pick the latest AWS Deep Learning Base GPU AMI (Ubuntu 22.04) — works fine on CPU instances too."
  type        = string
  default     = ""
}

variable "project_tag" {
  description = "Value for the Project tag, applied to all resources for cost-tracking."
  type        = string
  default     = "world-models-02c"
}

variable "extra_tags" {
  description = "Additional resource tags."
  type        = map(string)
  default     = {}
}
