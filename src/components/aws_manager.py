"""
aws_manager.py
AWS integration for omni-dev-agent
"""

import boto3
from botocore.exceptions import ClientError


class AWSManager:
    """
    Usage Example:
    --------------
    from aws_manager import AWSManager
    aws = AWSManager(region_name='us-east-1')

    # EC2
    ec2_config = {
        'ImageId': 'ami-12345678',
        'InstanceType': 't2.micro',
        'MinCount': 1,
        'MaxCount': 1
    }
    ec2_response = aws.deploy_ec2_instance(ec2_config)
    print(ec2_response)

    # Lambda
    lambda_config = {
        'FunctionName': 'my-function',
        'Runtime': 'python3.9',
        'Role': 'arn:aws:iam::123456789012:role/service-role/my-role',
        'Handler': 'lambda_function.lambda_handler',
        'Code': {'ZipFile': b'bytes_of_zip_file'}
    }
    lambda_response = aws.deploy_lambda_function(lambda_config)
    print(lambda_response)

    # ECS
    ecs_config = {
        'cluster': 'default',
        'serviceName': 'my-service',
        'taskDefinition': 'my-task:1',
        'desiredCount': 1
    }
    ecs_response = aws.deploy_ecs_service(ecs_config)
    print(ecs_response)

    # CodeDeploy
    codedeploy_config = {
        'applicationName': 'MyApp',
        'deploymentGroupName': 'MyDG',
        'revision': {
            'revisionType': 'S3',
            's3Location': {
                'bucket': 'my-bucket',
                'key': 'my-app.zip',
                'bundleType': 'zip'
            }
        }
    }
    codedeploy_response = aws.trigger_codedeploy_deployment(codedeploy_config)
    print(codedeploy_response)

    # Extra features
    aws.list_ec2_instances()
    aws.list_lambda_functions()
    aws.list_ecs_clusters()
    aws.list_codedeploy_apps()
    """

    def __init__(self, region_name="us-east-1"):
        self.region = region_name
        self.ec2 = boto3.client("ec2", region_name=self.region)
        self.lambda_client = boto3.client("lambda", region_name=self.region)
        self.ecs = boto3.client("ecs", region_name=self.region)
        self.codedeploy = boto3.client("codedeploy", region_name=self.region)

    def deploy_ec2_instance(self, config):
        try:
            response = self.ec2.run_instances(**config)
            return response
        except ClientError as e:
            print(f"EC2 Error: {e}")
            return None

    def list_ec2_instances(self):
        try:
            response = self.ec2.describe_instances()
            for reservation in response["Reservations"]:
                for instance in reservation["Instances"]:
                    print(
                        f"Instance ID: {instance['InstanceId']}, State: {instance['State']['Name']}"
                    )
        except ClientError as e:
            print(f"EC2 List Error: {e}")

    def deploy_lambda_function(self, config):
        try:
            response = self.lambda_client.create_function(**config)
            return response
        except ClientError as e:
            print(f"Lambda Error: {e}")
            return None

    def list_lambda_functions(self):
        try:
            response = self.lambda_client.list_functions()
            for fn in response["Functions"]:
                print(f"Function Name: {fn['FunctionName']}, Runtime: {fn['Runtime']}")
        except ClientError as e:
            print(f"Lambda List Error: {e}")

    def deploy_ecs_service(self, config):
        try:
            response = self.ecs.create_service(**config)
            return response
        except ClientError as e:
            print(f"ECS Error: {e}")
            return None

    def list_ecs_clusters(self):
        try:
            response = self.ecs.list_clusters()
            for arn in response["clusterArns"]:
                print(f"Cluster ARN: {arn}")
        except ClientError as e:
            print(f"ECS List Error: {e}")

    def trigger_codedeploy_deployment(self, config):
        try:
            response = self.codedeploy.create_deployment(**config)
            return response
        except ClientError as e:
            print(f"CodeDeploy Error: {e}")
            return None

    def list_codedeploy_apps(self):
        try:
            response = self.codedeploy.list_applications()
            for app in response["applications"]:
                print(f"CodeDeploy Application: {app}")
        except ClientError as e:
            print(f"CodeDeploy List Error: {e}")
