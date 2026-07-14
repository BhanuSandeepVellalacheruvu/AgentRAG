"""Live AWS Infrastructure Dashboard for AgentRAG.

Fetches real-time metrics from AWS CloudWatch, S3, DynamoDB,
Lambda, and Cost Explorer, then serves a beautiful dashboard UI.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import boto3
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

load_dotenv()

app = FastAPI(title="AgentRAG Live Dashboard")

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

BOTO_KWARGS: dict[str, Any] = {
    "region_name": AWS_REGION,
}
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    BOTO_KWARGS["aws_access_key_id"] = AWS_ACCESS_KEY_ID
    BOTO_KWARGS["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY


def _client(service: str) -> Any:
    return boto3.client(service, **BOTO_KWARGS)


def _resource(service: str) -> Any:
    return boto3.resource(service, **BOTO_KWARGS)


@app.get("/")
async def serve_dashboard() -> FileResponse:
    """Serve the dashboard HTML page."""
    return FileResponse(
        Path(__file__).parent / "index.html",
        media_type="text/html",
    )


@app.get("/api/lambda")
async def get_lambda_info() -> JSONResponse:
    """Fetch Lambda function details and recent invocation metrics."""
    try:
        lam = _client("lambda")
        cw = _client("cloudwatch")

        # Find our function
        funcs = lam.list_functions()["Functions"]
        agent_fn = None
        for f in funcs:
            if "AgentRag" in f.get("FunctionName", ""):
                agent_fn = f
                break

        if not agent_fn:
            return JSONResponse({"status": "not_deployed", "functions": []})

        fn_name = agent_fn["FunctionName"]
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=7)

        # Get invocation count (last 7 days)
        invocations = cw.get_metric_statistics(
            Namespace="AWS/Lambda",
            MetricName="Invocations",
            Dimensions=[{"Name": "FunctionName", "Value": fn_name}],
            StartTime=start,
            EndTime=now,
            Period=86400,
            Statistics=["Sum"],
        )

        # Get errors (last 7 days)
        errors = cw.get_metric_statistics(
            Namespace="AWS/Lambda",
            MetricName="Errors",
            Dimensions=[{"Name": "FunctionName", "Value": fn_name}],
            StartTime=start,
            EndTime=now,
            Period=86400,
            Statistics=["Sum"],
        )

        # Get average duration
        duration = cw.get_metric_statistics(
            Namespace="AWS/Lambda",
            MetricName="Duration",
            Dimensions=[{"Name": "FunctionName", "Value": fn_name}],
            StartTime=start,
            EndTime=now,
            Period=86400,
            Statistics=["Average"],
        )

        total_invocations = sum(
            dp["Sum"] for dp in invocations.get("Datapoints", [])
        )
        total_errors = sum(dp["Sum"] for dp in errors.get("Datapoints", []))
        avg_duration = 0.0
        dur_points = duration.get("Datapoints", [])
        if dur_points:
            avg_duration = sum(dp["Average"] for dp in dur_points) / len(
                dur_points
            )

        # Daily invocations for chart
        daily = []
        for dp in sorted(
            invocations.get("Datapoints", []), key=lambda x: x["Timestamp"]
        ):
            daily.append(
                {
                    "date": dp["Timestamp"].strftime("%m/%d"),
                    "count": int(dp["Sum"]),
                }
            )

        return JSONResponse(
            {
                "status": "deployed",
                "function_name": fn_name,
                "runtime": agent_fn.get("PackageType", "Image"),
                "memory_mb": agent_fn.get("MemorySize", 0),
                "timeout_s": agent_fn.get("Timeout", 0),
                "code_size_mb": round(
                    agent_fn.get("CodeSize", 0) / 1024 / 1024, 1
                ),
                "last_modified": agent_fn.get("LastModified", ""),
                "total_invocations_7d": int(total_invocations),
                "total_errors_7d": int(total_errors),
                "avg_duration_ms": round(avg_duration, 1),
                "daily_invocations": daily,
            }
        )
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)})


@app.get("/api/dynamodb")
async def get_dynamodb_info() -> JSONResponse:
    """Fetch DynamoDB table details and item counts."""
    try:
        ddb = _client("dynamodb")
        tables = ddb.list_tables().get("TableNames", [])

        our_tables = [t for t in tables if "agentrag" in t.lower()]
        result = []

        for table_name in our_tables:
            desc = ddb.describe_table(TableName=table_name)["Table"]
            result.append(
                {
                    "name": table_name,
                    "status": desc.get("TableStatus", "UNKNOWN"),
                    "item_count": desc.get("ItemCount", 0),
                    "size_bytes": desc.get("TableSizeBytes", 0),
                    "billing": desc.get("BillingModeSummary", {}).get(
                        "BillingMode", "PAY_PER_REQUEST"
                    ),
                    "key_schema": [
                        {
                            "name": k["AttributeName"],
                            "type": k["KeyType"],
                        }
                        for k in desc.get("KeySchema", [])
                    ],
                }
            )

        return JSONResponse({"status": "ok", "tables": result})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)})


@app.get("/api/s3")
async def get_s3_info() -> JSONResponse:
    """Fetch S3 bucket details and object counts."""
    try:
        s3 = _client("s3")
        buckets = s3.list_buckets().get("Buckets", [])
        our_buckets = [
            b for b in buckets if "agentrag" in b["Name"].lower()
        ]

        result = []
        for bucket in our_buckets:
            bucket_name = bucket["Name"]
            try:
                objects = s3.list_objects_v2(
                    Bucket=bucket_name, MaxKeys=1000
                )
                obj_count = objects.get("KeyCount", 0)
                total_size = sum(
                    o.get("Size", 0)
                    for o in objects.get("Contents", [])
                )
            except Exception:
                obj_count = 0
                total_size = 0

            result.append(
                {
                    "name": bucket_name,
                    "created": bucket["CreationDate"].isoformat(),
                    "object_count": obj_count,
                    "total_size_mb": round(total_size / 1024 / 1024, 2),
                }
            )

        return JSONResponse({"status": "ok", "buckets": result})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)})


@app.get("/api/apigateway")
async def get_apigateway_info() -> JSONResponse:
    """Fetch API Gateway details."""
    try:
        apigw = _client("apigatewayv2")
        apis = apigw.get_apis().get("Items", [])
        our_apis = [
            a for a in apis if "AgentRAG" in a.get("Name", "")
        ]

        result = []
        for api in our_apis:
            endpoint = api.get("ApiEndpoint", "N/A")
            result.append(
                {
                    "name": api.get("Name", ""),
                    "api_id": api.get("ApiId", ""),
                    "endpoint": endpoint,
                    "protocol": api.get("ProtocolType", ""),
                    "created": api.get("CreatedDate", "").isoformat()
                    if api.get("CreatedDate")
                    else "",
                }
            )

        return JSONResponse({"status": "ok", "apis": result})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)})


@app.get("/api/costs")
async def get_cost_info() -> JSONResponse:
    """Fetch current month cost from AWS Cost Explorer."""
    try:
        ce = _client("ce")
        now = datetime.now(timezone.utc)
        start_of_month = now.replace(day=1).strftime("%Y-%m-%d")
        today = now.strftime("%Y-%m-%d")

        cost = ce.get_cost_and_usage(
            TimePeriod={"Start": start_of_month, "End": today},
            Granularity="MONTHLY",
            Metrics=["UnblendedCost"],
            GroupBy=[
                {"Type": "DIMENSION", "Key": "SERVICE"},
            ],
        )

        services = []
        total = 0.0
        for group in cost.get("ResultsByTime", [{}])[0].get("Groups", []):
            amount = float(
                group["Metrics"]["UnblendedCost"]["Amount"]
            )
            svc_name = group["Keys"][0]
            total += amount
            if amount > 0.0001:
                services.append(
                    {"service": svc_name, "cost": round(amount, 4)}
                )

        services.sort(key=lambda x: x["cost"], reverse=True)

        return JSONResponse(
            {
                "status": "ok",
                "period": f"{start_of_month} to {today}",
                "total_cost": round(total, 4),
                "services": services,
            }
        )
    except Exception as e:
        return JSONResponse(
            {"status": "error", "error": str(e), "total_cost": 0}
        )


@app.get("/api/budget")
async def get_budget_info() -> JSONResponse:
    """Fetch AWS Budget status."""
    try:
        budgets_client = _client("budgets")
        sts = _client("sts")
        account_id = sts.get_caller_identity()["Account"]

        resp = budgets_client.describe_budgets(AccountId=account_id)
        budgets = resp.get("Budgets", [])

        result = []
        for b in budgets:
            limit = b.get("BudgetLimit", {})
            actual = b.get("CalculatedSpend", {}).get(
                "ActualSpend", {}
            )
            forecast = b.get("CalculatedSpend", {}).get(
                "ForecastedSpend", {}
            )
            result.append(
                {
                    "name": b.get("BudgetName", ""),
                    "type": b.get("BudgetType", ""),
                    "limit": f"${limit.get('Amount', '0')}",
                    "actual_spend": f"${actual.get('Amount', '0')}",
                    "forecasted_spend": f"${forecast.get('Amount', '0')}"
                    if forecast
                    else "N/A",
                    "time_unit": b.get("TimeUnit", ""),
                }
            )

        return JSONResponse({"status": "ok", "budgets": result})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)})


@app.get("/api/cloudformation")
async def get_stack_info() -> JSONResponse:
    """Fetch CloudFormation stack status."""
    try:
        cfn = _client("cloudformation")
        stacks = cfn.describe_stacks().get("Stacks", [])
        our_stacks = [
            s
            for s in stacks
            if "AgentRag" in s.get("StackName", "")
            or "CDKToolkit" in s.get("StackName", "")
        ]

        result = []
        for s in our_stacks:
            outputs = {
                o["OutputKey"]: o["OutputValue"]
                for o in s.get("Outputs", [])
            }
            result.append(
                {
                    "name": s["StackName"],
                    "status": s.get("StackStatus", ""),
                    "created": s.get("CreationTime", "").isoformat()
                    if s.get("CreationTime")
                    else "",
                    "updated": s.get("LastUpdatedTime", "").isoformat()
                    if s.get("LastUpdatedTime")
                    else "",
                    "outputs": outputs,
                }
            )

        return JSONResponse({"status": "ok", "stacks": result})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8501)
