from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    aws_region: str
    bedrock_model_id: str
    s3_bucket_name: str
    dynamodb_table_name: str
    budget_alarm_threshold: float

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

def get_settings() -> Settings:
    return Settings() # type: ignore
