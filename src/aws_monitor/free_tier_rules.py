# This file defines the Free Tier rules for various AWS services.
# It will serve as the "Free Tier Rules Database" for the monitoring agent.

FREE_TIER_RULES = {
    "ec2": {
        "t2.micro": {
            "limit": 750,  # hours per month
            "unit": "hours",
            "data_source": "cloudwatch",  # or "api" for direct API calls
            "metrics": [
                {
                    "Namespace": "AWS/EC2",
                    "MetricName": "CPUUtilization",
                    "Statistic": "Average",
                }
            ],
            "conditions": {
                "instance_type": "t2.micro",
                "os": [
                    "linux",
                    "windows",
                ],  # Free tier applies to both Linux and Windows t2.micro
            },
        },
        # Add other EC2 instance types or services as needed
    },
    # Add other AWS services here (e.g., "s3", "lambda", "rds")
}
