#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# auto_stop_ec2.sh
# Run this at the END of your training job (on the EC2 instance).
# It stops the instance itself after training completes or fails.
#
# Usage: Add to GitHub Actions after training step:
#   ssh ubuntu@EC2 "bash ~/image-mlops/scripts/auto_stop_ec2.sh"
#
# Or call it at the end of train.py using subprocess.
# ─────────────────────────────────────────────────────────────────

set -e

INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)

echo "Training complete. Stopping instance $INSTANCE_ID in $REGION ..."

# Give a few seconds for logs to flush
sleep 5

# Stop the instance (NOT terminate — your data is safe)
aws ec2 stop-instances \
  --instance-ids "$INSTANCE_ID" \
  --region "$REGION"

echo "Stop command sent. Instance will stop in ~30 seconds."
