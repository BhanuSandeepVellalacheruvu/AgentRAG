from typing import Any

import aws_cdk as cdk
from aws_cdk import (
    RemovalPolicy,
    Stack,
)
from aws_cdk import (
    aws_apigatewayv2 as apigw,
)
from aws_cdk import (
    aws_apigatewayv2_integrations as integrations,
)
from aws_cdk import (
    aws_budgets as budgets,
)
from aws_cdk import (
    aws_dynamodb as dynamodb,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_lambda as lm,
)
from aws_cdk import (
    aws_s3 as s3,
)
from constructs import Construct

from agentrag.config import get_settings


class AgentRagStack(Stack):
    """CDK Stack to deploy AgentRAG on AWS."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs: Any) -> None:
        """Initialize the stack."""
        super().__init__(scope, construct_id, **kwargs)

        settings = get_settings()

        # 1. S3 Bucket for documents and vector index
        bucket = s3.Bucket(
            self,
            "AgentRagBucket",
            bucket_name=settings.s3_bucket_name,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
        )

        # 2. DynamoDB Table for traces
        traces_table = dynamodb.Table(
            self,
            "AgentRagTracesTable",
            table_name=settings.dynamodb_table_name,
            partition_key=dynamodb.Attribute(
                name="id", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # 3. DynamoDB Table for index metadata/integrity validation (SEC-003)
        metadata_table = dynamodb.Table(
            self,
            "AgentRagMetadataTable",
            table_name="agentrag-index-metadata",
            partition_key=dynamodb.Attribute(
                name="index_id", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # 4. Lambda Function running FastAPI app in Docker container
        lambda_fn = lm.DockerImageFunction(
            self,
            "AgentRagLambda",
            code=lm.DockerImageCode.from_image_asset(
                directory="../../../",  # Root directory with Dockerfile
                exclude=[
                    ".venv",
                    ".git",
                    ".github",
                    "src/agentrag/infra/cdk.out",
                    "data/index",
                ],
            ),
            environment={
                "BEDROCK_MODEL_ID": settings.bedrock_model_id,
                "BEDROCK_EMBEDDING_MODEL_ID": settings.bedrock_embedding_model_id,
                "S3_BUCKET_NAME": bucket.bucket_name,
                "DYNAMODB_TABLE_NAME": traces_table.table_name,
                "API_KEY": settings.api_key,
                "USE_MOCK_LLM": "false",
                "USE_MOCK_EMBEDDINGS": "false",
            },
            timeout=cdk.Duration.seconds(60),
            memory_size=1024,
        )

        # 5. Grant read/write access to S3 and DynamoDB resources
        bucket.grant_read_write(lambda_fn)
        traces_table.grant_read_write_data(lambda_fn)
        metadata_table.grant_read_write_data(lambda_fn)

        # 6. Bedrock invoke permissions
        lambda_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=[
                    f"arn:aws:bedrock:{settings.aws_region}::foundation-model/*"
                ],
            )
        )

        # 7. API Gateway HTTP API for low-cost low-latency routing
        http_api = apigw.HttpApi(
            self,
            "AgentRagApi",
            api_name="AgentRAG API Gateway Proxy",
            default_integration=integrations.HttpLambdaIntegration(
                "AgentRagIntegration", lambda_fn
            ),
        )

        # 8. Budget alarm to prevent cost overrun
        budgets.CfnBudget(
            self,
            "AgentRagBudget",
            budget=budgets.CfnBudget.BudgetDataProperty(
                budget_name="AgentRagMonthlyBudget",
                budget_type="COST",
                budget_limit=budgets.CfnBudget.SpendProperty(
                    amount=settings.budget_alarm_threshold, unit="USD"
                ),
                time_unit="MONTHLY",
            ),
            notifications_with_subscribers=[
                budgets.CfnBudget.NotificationWithSubscribersProperty(
                    notification=budgets.CfnBudget.NotificationProperty(
                        comparison_operator="GREATER_THAN",
                        notification_type="ACTUAL",
                        threshold=100,
                        threshold_type="PERCENTAGE",
                    ),
                    subscribers=[
                        budgets.CfnBudget.SubscriberProperty(
                            address="dummy@example.com",
                            subscription_type="EMAIL",
                        )
                    ],
                )
            ],
        )

        # 9. CloudFormation Outputs
        cdk.CfnOutput(
            self,
            "ApiUrl",
            value=http_api.url or "URL_NOT_FOUND",
            description="The URL of the API Gateway proxy",
        )
