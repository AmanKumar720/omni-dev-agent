
import boto3
from datetime import datetime, timedelta
from .free_tier_rules import FREE_TIER_RULES

class AWSFreeTierMonitor:
    def __init__(self):
        self.ec2_client = boto3.client('ec2', region_name='us-east-1')
        self.cloudwatch_client = boto3.client('cloudwatch', region_name='us-east-1')

    def get_ec2_free_tier_usage(self):
        free_tier_ec2_rules = FREE_TIER_RULES.get('ec2', {})
        if not free_tier_ec2_rules:
            print("No EC2 Free Tier rules defined.")
            return {}

        usage_data = {}
        try:
            # Get all running instances
            response = self.ec2_client.describe_instances(
                Filters=[
                    {'Name': 'instance-state-name', 'Values': ['running']}
                ]
            )

            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    instance_id = instance['InstanceId']
                    instance_type = instance['InstanceType']
                    platform = instance.get('PlatformDetails', 'Linux/UNIX') # Default to Linux/UNIX

                    # Check if this instance type is in our Free Tier rules
                    if instance_type in free_tier_ec2_rules:
                        rule = free_tier_ec2_rules[instance_type]
                        # Basic check for OS condition (can be expanded)
                        if 'os' in rule['conditions'] and platform.lower() not in [o.lower() for o in rule['conditions']['os']]:
                            continue # Skip if OS doesn't match free tier condition

                        print(f"Monitoring {instance_type} instance {instance_id} ({platform})")

                        # For simplicity, we'll just count running instances for now.
                        # Actual hour calculation would involve more complex logic with CloudWatch metrics
                        # or instance launch/stop times.
                        # This is a placeholder for actual usage calculation.
                        usage_data[instance_id] = {
                            'instance_type': instance_type,
                            'platform': platform,
                            'estimated_monthly_hours': 720 # Placeholder: assuming always running for a month
                        }

        except Exception as e:
            print(f"Error fetching EC2 instances: {e}")
        return usage_data

    def monitor_all_services(self):
        print("Starting AWS Free Tier monitoring...")
        ec2_usage = self.get_ec2_free_tier_usage()
        print("EC2 Usage:", ec2_usage)
        # Add calls for other services here

if __name__ == "__main__":
    monitor = AWSFreeTierMonitor()
    monitor.monitor_all_services()
