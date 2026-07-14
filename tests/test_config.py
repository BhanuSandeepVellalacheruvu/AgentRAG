import os

from agentrag.common.config import get_settings


def test_get_settings_loads_env() -> None:
    os.environ["AWS_REGION"] = "us-west-2"
    os.environ["BEDROCK_MODEL_ID"] = "dummy-model"
    os.environ["S3_BUCKET_NAME"] = "test-bucket"
    os.environ["DYNAMODB_TABLE_NAME"] = "test-table"
    os.environ["BUDGET_ALARM_THRESHOLD"] = "2.0"

    settings = get_settings()
    assert settings.aws_region == "us-west-2"
    assert settings.bedrock_model_id == "dummy-model"
    assert settings.s3_bucket_name == "test-bucket"
    assert settings.dynamodb_table_name == "test-table"
    assert settings.budget_alarm_threshold == 2.0
