#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# setup_billing_alerts.sh
# Sets up AWS Budget + CloudWatch billing alarms so you get
# an email BEFORE your bill gets too high.
#
# Run this ONCE from your local machine (not EC2).
# Replace placeholders before running.
#
# Usage:
#   chmod +x setup_billing_alerts.sh
#   ./setup_billing_alerts.sh
# ─────────────────────────────────────────────────────────────────

set -e

# ── CONFIGURE THESE ───────────────────────────────────────────────
YOUR_EMAIL="your-email@gmail.com"       # Where alerts are sent
ACCOUNT_ID="123456789012"               # Your 12-digit AWS account ID
MONTHLY_BUDGET_USD="30"                 # Your monthly budget limit
ALERT_AT_PERCENT_50="15"               # Alert at 50% of budget ($15)
ALERT_AT_PERCENT_80="24"               # Alert at 80% of budget ($24)
REGION="us-east-1"
# ─────────────────────────────────────────────────────────────────

echo "Setting up AWS billing alerts for account $ACCOUNT_ID..."
echo "Monthly budget: \$$MONTHLY_BUDGET_USD"
echo "Alert email: $YOUR_EMAIL"
echo ""

# ── Step 1: SNS Topic for email alerts ───────────────────────────
echo "[1/4] Creating SNS topic for billing alerts..."
TOPIC_ARN=$(aws sns create-topic \
  --name "aws-billing-alerts" \
  --region us-east-1 \
  --query 'TopicArn' \
  --output text)
echo "SNS Topic: $TOPIC_ARN"

# Subscribe your email
aws sns subscribe \
  --topic-arn "$TOPIC_ARN" \
  --protocol email \
  --notification-endpoint "$YOUR_EMAIL" \
  --region us-east-1

echo "CHECK YOUR EMAIL and confirm the SNS subscription before continuing!"
read -p "Press Enter once you have confirmed the email subscription..."

# ── Step 2: CloudWatch Billing Alarm ($10 threshold) ─────────────
echo "[2/4] Creating CloudWatch billing alarm (\$10)..."
aws cloudwatch put-metric-alarm \
  --alarm-name "billing-alert-10usd" \
  --alarm-description "Alert when bill exceeds \$10" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --statistic Maximum \
  --period 86400 \
  --threshold 10 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=Currency,Value=USD \
  --evaluation-periods 1 \
  --alarm-actions "$TOPIC_ARN" \
  --region us-east-1
echo "CloudWatch alarm created: billing-alert-10usd"

# Another alarm at your full budget
aws cloudwatch put-metric-alarm \
  --alarm-name "billing-alert-${MONTHLY_BUDGET_USD}usd" \
  --alarm-description "Alert when bill hits budget of \$$MONTHLY_BUDGET_USD" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --statistic Maximum \
  --period 86400 \
  --threshold "$MONTHLY_BUDGET_USD" \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=Currency,Value=USD \
  --evaluation-periods 1 \
  --alarm-actions "$TOPIC_ARN" \
  --region us-east-1
echo "CloudWatch alarm created: billing-alert-${MONTHLY_BUDGET_USD}usd"

# ── Step 3: AWS Budget with forecast alert ────────────────────────
echo "[3/4] Creating AWS Budget with forecast alert..."
cat > /tmp/budget.json << EOF
{
  "BudgetName": "MLOps-Monthly-Budget",
  "BudgetLimit": {
    "Amount": "$MONTHLY_BUDGET_USD",
    "Unit": "USD"
  },
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST"
}
EOF

cat > /tmp/notifications.json << EOF
[
  {
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 50,
      "ThresholdType": "PERCENTAGE",
      "NotificationState": "ALARM"
    },
    "Subscribers": [
      { "SubscriptionType": "EMAIL", "Address": "$YOUR_EMAIL" }
    ]
  },
  {
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 80,
      "ThresholdType": "PERCENTAGE",
      "NotificationState": "ALARM"
    },
    "Subscribers": [
      { "SubscriptionType": "EMAIL", "Address": "$YOUR_EMAIL" }
    ]
  },
  {
    "Notification": {
      "NotificationType": "FORECASTED",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 100,
      "ThresholdType": "PERCENTAGE",
      "NotificationState": "ALARM"
    },
    "Subscribers": [
      { "SubscriptionType": "EMAIL", "Address": "$YOUR_EMAIL" }
    ]
  }
]
EOF

aws budgets create-budget \
  --account-id "$ACCOUNT_ID" \
  --budget file:///tmp/budget.json \
  --notifications-with-subscribers file:///tmp/notifications.json

echo "AWS Budget created: MLOps-Monthly-Budget (\$$MONTHLY_BUDGET_USD/month)"

# ── Step 4: EC2 Instance Scheduler (auto-stop idle instance) ──────
echo "[4/4] Creating Lambda to auto-stop EC2 if idle for 2 hours..."
cat > /tmp/stop_idle_ec2.py << 'PYEOF'
import boto3, json
from datetime import datetime, timezone, timedelta

def lambda_handler(event, context):
    """Stop EC2 instances that have been running for over 2 hours with low CPU."""
    ec2     = boto3.client('ec2', region_name='us-east-1')
    cw      = boto3.client('cloudwatch', region_name='us-east-1')
    cutoff  = datetime.now(timezone.utc) - timedelta(hours=2)

    response = ec2.describe_instances(
        Filters=[
            {'Name': 'tag:Name',    'Values': ['mlops-trainer']},
            {'Name': 'instance-state-name', 'Values': ['running']}
        ]
    )

    stopped = []
    for r in response['Reservations']:
        for inst in r['Instances']:
            iid        = inst['InstanceId']
            launch_t   = inst['LaunchTime']

            # Only check instances running for > 2 hours
            if launch_t > cutoff:
                continue

            # Check average CPU utilisation over last 30 minutes
            metrics = cw.get_metric_statistics(
                Namespace='AWS/EC2', MetricName='CPUUtilization',
                Dimensions=[{'Name': 'InstanceId', 'Value': iid}],
                StartTime=datetime.now(timezone.utc) - timedelta(minutes=30),
                EndTime=datetime.now(timezone.utc),
                Period=1800, Statistics=['Average']
            )

            if metrics['Datapoints']:
                avg_cpu = metrics['Datapoints'][0]['Average']
                if avg_cpu < 5.0:   # Less than 5% CPU = idle
                    print(f"Stopping idle instance {iid} (CPU: {avg_cpu:.1f}%)")
                    ec2.stop_instances(InstanceIds=[iid])
                    stopped.append(iid)

    return {'stopped': stopped, 'count': len(stopped)}
PYEOF

echo ""
echo "========================================"
echo "  Billing protection setup complete!"
echo ""
echo "  You will receive email alerts when:"
echo "  - Bill exceeds \$10"
echo "  - Bill exceeds 50% of budget (\$$ALERT_AT_PERCENT_50)"
echo "  - Bill exceeds 80% of budget (\$$ALERT_AT_PERCENT_80)"
echo "  - Bill is FORECASTED to exceed \$$MONTHLY_BUDGET_USD"
echo ""
echo "  Lambda idle-stopper code saved to:"
echo "  /tmp/stop_idle_ec2.py (deploy manually via AWS Console)"
echo "========================================"
