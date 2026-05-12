output "instance_id" {
  description = "EC2 instance ID."
  value       = aws_instance.experiment.id
}

output "public_ip" {
  description = "Public IPv4 address."
  value       = aws_instance.experiment.public_ip
}

output "public_dns" {
  description = "Public DNS name."
  value       = aws_instance.experiment.public_dns
}

output "ssh_command" {
  description = "Ready-to-paste SSH command (uses your local ~/.ssh/<key_name>.pem; adjust if your key path differs)."
  value       = "ssh -i ~/.ssh/${var.key_name}.pem ubuntu@${aws_instance.experiment.public_ip}"
}

output "rsync_command" {
  description = "Push the 02c project up to the instance. Run from the repo root."
  value       = "rsync -avz --exclude='__pycache__' --exclude='.venv' --exclude='data/' --exclude='checkpoints/' -e 'ssh -i ~/.ssh/${var.key_name}.pem' 02c-distill-scaled/ ubuntu@${aws_instance.experiment.public_ip}:~/02c-distill-scaled/"
}

output "instance_type_chosen" {
  description = "Echo of the instance type provisioned, for sanity."
  value       = var.instance_type
}

output "billing_mode" {
  description = "Whether this is a spot or on-demand instance."
  value       = var.use_spot ? "spot" : "on-demand"
}

output "destroy_command" {
  description = "Remember to tear it down when done. Costs accrue per second the instance exists."
  value       = "terraform destroy -auto-approve"
}
