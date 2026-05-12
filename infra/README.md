# Terraform — AWS box for the 02c distillation experiment

Spins up one EC2 instance, installs Stockfish + `uv`, opens SSH, and
outputs paste-ready commands for pushing the code and tearing it down.
Designed for ephemeral one-shot experiment runs.

## Files

| File | What |
|---|---|
| `main.tf` | Provider, AMI lookup, security group, EC2 instance |
| `variables.tf` | Every knob, with comments. The important one: `instance_type` |
| `outputs.tf` | SSH command, rsync command, destroy reminder |
| `user_data.sh` | First-boot bootstrap (installs stockfish, uv) |

## Quick start

```bash
# 0. One-time: AWS creds in your env (or ~/.aws/credentials)
export AWS_PROFILE=your-profile
# Or: export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...

# 1. One-time: create an SSH key pair in the AWS region you'll use
aws ec2 create-key-pair \
  --region us-east-1 \
  --key-name world-models-02c \
  --query KeyMaterial --output text > ~/.ssh/world-models-02c.pem
chmod 600 ~/.ssh/world-models-02c.pem

# 2. Init Terraform
cd infra
terraform init

# 3. Provision (spot g5.24xlarge by default)
terraform apply \
  -var="key_name=world-models-02c" \
  -var="allowed_ssh_cidr=$(curl -s ifconfig.me)/32"

# 4. Grab the commands Terraform printed
terraform output ssh_command
terraform output rsync_command

# 5. Push code up
$(terraform output -raw rsync_command)

# 6. SSH in
$(terraform output -raw ssh_command)

# On the instance:
cd ~/02c-distill-scaled
uv sync
nohup bash scripts/run_overnight.sh > /tmp/run.log 2>&1 &
disown
tail -f /tmp/run.log

# 7. When done, pull results back
scp -i ~/.ssh/world-models-02c.pem \
  ubuntu@<PUBLIC_IP>:'~/02c-distill-scaled/checkpoints/run01/*' \
  ./02c-distill-scaled/checkpoints/run01/

# 8. !!! TEAR IT DOWN !!! (you're billed every second)
terraform destroy -auto-approve
```

## Picking an `instance_type`

The variable is documented inline in `variables.tf`. Quick guide:

| Goal | `instance_type` | Approx. spot $/h | Full pipeline wall |
|---|---|---:|---|
| Cheapest fast (CPU only) | `c7i.16xlarge` | $0.80 | ~2h |
| **Sweet spot (default)** | **`g5.24xlarge`** | **$2.50** | **~1h** |
| Absolute fastest single-box | `g5.48xlarge` | $5 | ~40 min |
| Overkill | `p4d.24xlarge` | $10–15 | ~30 min (wastes 7 GPUs) |

Pricing is approximate, varies by region. Check current prices with:
```bash
aws ec2 describe-spot-price-history \
  --instance-types g5.24xlarge \
  --product-descriptions "Linux/UNIX" \
  --max-items 1 \
  --query 'SpotPriceHistory[0].SpotPrice'
```

## Cost guardrails

- **Default is spot.** ~70% cheaper than on-demand, can be interrupted
  with a 2-minute warning. For an 8-hour experiment that's acceptable risk;
  for an irreplaceable run, set `-var="use_spot=false"`.
- **Default security group is open to 0.0.0.0/0 SSH.** Always override
  with `-var="allowed_ssh_cidr=YOUR_IP/32"`. Example above uses
  `ifconfig.me` to auto-fill.
- **Always `terraform destroy`** after the experiment. A forgotten
  `g5.24xlarge` running 24/7 on-demand is ~$200/day.

## What's NOT included (intentionally)

- **No S3 backend for state.** Local `terraform.tfstate` is fine for a
  one-off experiment. If you'll run many experiments, add an S3+DynamoDB
  backend.
- **No VPC, no NAT gateway, no IAM beyond the default.** Uses the default
  VPC. Don't run this in your production AWS account — make a dedicated
  sandbox account.
- **No autoscaling / no second instance.** One box, one experiment.
- **Code is NOT cloned by user_data.** `rsync` it after the box is up so
  you can iterate locally on the script without re-provisioning.

## Common pitfalls

- **`key_name` must already exist in the region.** Terraform won't create
  it for you; that's a separate AWS resource type and we keep secrets out
  of git.
- **Spot can fail if capacity is constrained.** If `terraform apply` errors
  with `InsufficientInstanceCapacity`, either retry, switch region, or
  flip `use_spot=false`.
- **Initial boot can take 3–5 minutes** before SSH responds (DLAMI is large).
  Wait for `/home/ubuntu/BOOTSTRAP_COMPLETE` to appear before assuming
  Stockfish/uv are installed.
