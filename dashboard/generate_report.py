"""Generate a live AWS dashboard as a Markdown file.

This script fetches real-time metrics from AWS and writes
a formatted DASHBOARD.md that renders beautifully on GitHub.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

import boto3


def client(service: str) -> Any:
    kwargs: dict[str, Any] = {
        "region_name": os.getenv("AWS_REGION", "us-east-1"),
    }
    key = os.getenv("AWS_ACCESS_KEY_ID")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    if key and secret:
        kwargs["aws_access_key_id"] = key
        kwargs["aws_secret_access_key"] = secret
    return boto3.client(service, **kwargs)


def get_lambda_info() -> dict[str, Any]:
    try:
        lam = client("lambda")
        cw = client("cloudwatch")
        funcs = lam.list_functions()["Functions"]
        fn = next((f for f in funcs if "AgentRag" in f.get("FunctionName", "")), None)
        if not fn:
            return {"status": "not_deployed"}

        name = fn["FunctionName"]
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=7)

        inv = cw.get_metric_statistics(
            Namespace="AWS/Lambda", MetricName="Invocations",
            Dimensions=[{"Name": "FunctionName", "Value": name}],
            StartTime=start, EndTime=now, Period=604800, Statistics=["Sum"],
        )
        err = cw.get_metric_statistics(
            Namespace="AWS/Lambda", MetricName="Errors",
            Dimensions=[{"Name": "FunctionName", "Value": name}],
            StartTime=start, EndTime=now, Period=604800, Statistics=["Sum"],
        )
        dur = cw.get_metric_statistics(
            Namespace="AWS/Lambda", MetricName="Duration",
            Dimensions=[{"Name": "FunctionName", "Value": name}],
            StartTime=start, EndTime=now, Period=604800, Statistics=["Average"],
        )

        total_inv = sum(dp["Sum"] for dp in inv.get("Datapoints", []))
        total_err = sum(dp["Sum"] for dp in err.get("Datapoints", []))
        avg_dur = 0.0
        pts = dur.get("Datapoints", [])
        if pts:
            avg_dur = sum(dp["Average"] for dp in pts) / len(pts)

        return {
            "status": "deployed",
            "name": name,
            "runtime": fn.get("PackageType", "Image"),
            "memory": fn.get("MemorySize", 0),
            "timeout": fn.get("Timeout", 0),
            "size_mb": round(fn.get("CodeSize", 0) / 1024 / 1024, 1),
            "last_modified": fn.get("LastModified", ""),
            "invocations_7d": int(total_inv),
            "errors_7d": int(total_err),
            "avg_duration_ms": round(avg_dur, 1),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_dynamodb_info() -> list[dict[str, Any]]:
    try:
        ddb = client("dynamodb")
        tables = ddb.list_tables().get("TableNames", [])
        result = []
        for t in tables:
            if "agentrag" not in t.lower():
                continue
            desc = ddb.describe_table(TableName=t)["Table"]
            result.append({
                "name": t,
                "status": desc.get("TableStatus", "UNKNOWN"),
                "items": desc.get("ItemCount", 0),
                "size_kb": round(desc.get("TableSizeBytes", 0) / 1024, 1),
                "billing": desc.get("BillingModeSummary", {}).get("BillingMode", "PAY_PER_REQUEST"),
            })
        return result
    except Exception as e:
        return [{"name": "error", "status": str(e), "items": 0, "size_kb": 0, "billing": ""}]


def get_s3_info() -> list[dict[str, Any]]:
    try:
        s3 = client("s3")
        buckets = s3.list_buckets().get("Buckets", [])
        result = []
        for b in buckets:
            if "agentrag" not in b["Name"].lower():
                continue
            try:
                objs = s3.list_objects_v2(Bucket=b["Name"], MaxKeys=1000)
                count = objs.get("KeyCount", 0)
                size = sum(o.get("Size", 0) for o in objs.get("Contents", []))
            except Exception:
                count, size = 0, 0
            result.append({
                "name": b["Name"],
                "objects": count,
                "size_mb": round(size / 1024 / 1024, 2),
            })
        return result
    except Exception as e:
        return [{"name": "error", "objects": 0, "size_mb": 0}]


def get_apigateway_info() -> dict[str, Any]:
    try:
        apigw = client("apigatewayv2")
        apis = apigw.get_apis().get("Items", [])
        api = next((a for a in apis if "AgentRAG" in a.get("Name", "")), None)
        if not api:
            return {"status": "not_found"}
        return {
            "status": "active",
            "name": api.get("Name", ""),
            "endpoint": api.get("ApiEndpoint", ""),
            "protocol": api.get("ProtocolType", ""),
            "api_id": api.get("ApiId", ""),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_cost_info() -> dict[str, Any]:
    try:
        ce = client("ce")
        now = datetime.now(timezone.utc)
        start = now.replace(day=1).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")
        cost = ce.get_cost_and_usage(
            TimePeriod={"Start": start, "End": end},
            Granularity="MONTHLY", Metrics=["UnblendedCost"],
            GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
        )
        services = []
        total = 0.0
        for g in cost.get("ResultsByTime", [{}])[0].get("Groups", []):
            amt = float(g["Metrics"]["UnblendedCost"]["Amount"])
            total += amt
            if amt > 0.0001:
                services.append({"service": g["Keys"][0], "cost": round(amt, 4)})
        services.sort(key=lambda x: x["cost"], reverse=True)
        return {"total": round(total, 4), "services": services, "period": f"{start} → {end}"}
    except Exception as e:
        return {"total": 0, "services": [], "period": "", "error": str(e)}


def get_budget_info() -> dict[str, Any]:
    try:
        budgets = client("budgets")
        sts = client("sts")
        acct = sts.get_caller_identity()["Account"]
        resp = budgets.describe_budgets(AccountId=acct)
        b = resp.get("Budgets", [{}])[0] if resp.get("Budgets") else {}
        return {
            "name": b.get("BudgetName", "N/A"),
            "limit": b.get("BudgetLimit", {}).get("Amount", "0"),
            "actual": b.get("CalculatedSpend", {}).get("ActualSpend", {}).get("Amount", "0"),
            "forecast": b.get("CalculatedSpend", {}).get("ForecastedSpend", {}).get("Amount", "0"),
        }
    except Exception as e:
        return {"name": "N/A", "limit": "0.50", "actual": "0", "forecast": "0", "error": str(e)}


def get_cfn_info() -> list[dict[str, Any]]:
    try:
        cfn = client("cloudformation")
        stacks = cfn.describe_stacks().get("Stacks", [])
        result = []
        for s in stacks:
            if "AgentRag" not in s.get("StackName", "") and "CDKToolkit" not in s.get("StackName", ""):
                continue
            outputs = {o["OutputKey"]: o["OutputValue"] for o in s.get("Outputs", [])}
            result.append({
                "name": s["StackName"],
                "status": s.get("StackStatus", ""),
                "created": s.get("CreationTime", "").isoformat() if s.get("CreationTime") else "",
                "outputs": outputs,
            })
        return result
    except Exception as e:
        return [{"name": "error", "status": str(e), "created": "", "outputs": {}}]


def generate_markdown() -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lam = get_lambda_info()
    ddb = get_dynamodb_info()
    s3 = get_s3_info()
    apigw = get_apigateway_info()
    costs = get_cost_info()
    budget = get_budget_info()
    cfn = get_cfn_info()

    # Build markdown
    md = []
    md.append("# ☁️ AgentRAG — Live AWS Infrastructure Dashboard")
    md.append("")
    md.append(f"> **Last updated:** {now}  ")
    md.append(f"> **AWS Account:** `454837411571` · **Region:** `us-east-1`  ")
    md.append(f"> *Auto-generated weekly by GitHub Actions*")
    md.append("")

    # Summary stats
    inv = lam.get("invocations_7d", 0) if lam.get("status") == "deployed" else "—"
    errs = lam.get("errors_7d", 0) if lam.get("status") == "deployed" else "—"
    md.append("## 📊 Summary")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| Lambda Invocations (7d) | **{inv}** |")
    md.append(f"| Lambda Errors (7d) | **{errs}** |")
    md.append(f"| Current Month Cost | **${costs.get('total', 0):.4f}** |")
    md.append(f"| Budget Limit | **${budget.get('limit', '0.50')}** |")
    md.append(f"| Budget Actual Spend | **${budget.get('actual', '0')}** |")

    main_stack = next((s for s in cfn if "AgentRag" in s["name"]), None)
    stack_status = main_stack["status"] if main_stack else "NOT DEPLOYED"
    md.append(f"| Stack Status | **{stack_status}** |")
    md.append("")

    # API Gateway
    md.append("## 🌐 API Gateway")
    md.append("")
    if apigw.get("status") == "active":
        md.append(f"| Property | Value |")
        md.append(f"|----------|-------|")
        md.append(f"| Name | `{apigw['name']}` |")
        md.append(f"| Endpoint | [{apigw['endpoint']}]({apigw['endpoint']}) |")
        md.append(f"| Protocol | `{apigw['protocol']}` |")
        md.append(f"| API ID | `{apigw['api_id']}` |")
    else:
        md.append(f"> ⚠️ Not deployed yet. Status: `{apigw.get('status', 'unknown')}`")
        if apigw.get("error"):
            md.append(f"> Error: `{apigw['error']}`")
    md.append("")

    # Lambda
    md.append("## ⚡ Lambda Function")
    md.append("")
    if lam.get("status") == "deployed":
        md.append("| Property | Value |")
        md.append("|----------|-------|")
        md.append(f"| Function Name | `{lam['name']}` |")
        md.append(f"| Runtime | `{lam['runtime']}` |")
        md.append(f"| Memory | `{lam['memory']} MB` |")
        md.append(f"| Timeout | `{lam['timeout']}s` |")
        md.append(f"| Image Size | `{lam['size_mb']} MB` |")
        md.append(f"| Avg Duration (7d) | `{lam['avg_duration_ms']} ms` |")
        md.append(f"| Invocations (7d) | **{lam['invocations_7d']}** |")
        md.append(f"| Errors (7d) | **{lam['errors_7d']}** |")
        md.append(f"| Last Modified | `{lam['last_modified']}` |")
    elif lam.get("status") == "not_deployed":
        md.append("> ⚠️ No AgentRag Lambda function found in this account/region.")
    else:
        md.append(f"> ❌ Error fetching Lambda info: `{lam.get('error', 'unknown')}`")
    md.append("")

    # S3
    md.append("## 🪣 S3 Buckets")
    md.append("")
    md.append("| Bucket | Objects | Size |")
    md.append("|--------|---------|------|")
    for b in s3:
        if b["name"] == "error":
            md.append(f"| ❌ Error | — | — |")
        else:
            md.append(f"| `{b['name']}` | {b['objects']} | {b['size_mb']} MB |")
    md.append("")

    # DynamoDB
    md.append("## 🗃️ DynamoDB Tables")
    md.append("")
    md.append("| Table | Status | Items | Size | Billing |")
    md.append("|-------|--------|-------|------|---------|")
    for t in ddb:
        status_icon = "🟢" if t["status"] == "ACTIVE" else "🟡"
        md.append(f"| `{t['name']}` | {status_icon} {t['status']} | {t['items']} | {t['size_kb']} KB | `{t['billing']}` |")
    md.append("")

    # CloudFormation
    md.append("## 📋 CloudFormation Stacks")
    md.append("")
    md.append("| Stack | Status | Created |")
    md.append("|-------|--------|---------|")
    for s in cfn:
        icon = "🟢" if "COMPLETE" in s["status"] else "🟡" if "PROGRESS" in s["status"] else "🔴"
        created = s["created"][:10] if s["created"] else "—"
        md.append(f"| `{s['name']}` | {icon} `{s['status']}` | {created} |")

    if main_stack and main_stack.get("outputs"):
        md.append("")
        md.append("**Stack Outputs:**")
        for k, v in main_stack["outputs"].items():
            md.append(f"- **{k}**: `{v}`")
    md.append("")

    # Costs
    md.append("## 💵 Cost Breakdown (Current Month)")
    md.append("")
    if costs.get("services"):
        md.append(f"*Period: {costs.get('period', '')}*")
        md.append("")
        md.append("| Service | Cost (USD) |")
        md.append("|---------|-----------|")
        for s in costs["services"]:
            md.append(f"| {s['service']} | ${s['cost']:.4f} |")
        md.append(f"| **Total** | **${costs['total']:.4f}** |")
    else:
        md.append("> No cost data available yet for this billing period.")
        if costs.get("error"):
            md.append(f"> Error: `{costs['error']}`")
    md.append("")

    # Budget
    md.append("## 💰 Budget")
    md.append("")
    md.append("| Property | Value |")
    md.append("|----------|-------|")
    md.append(f"| Budget Name | `{budget.get('name', 'N/A')}` |")
    md.append(f"| Monthly Limit | **${budget.get('limit', '0.50')}** |")
    md.append(f"| Actual Spend | **${budget.get('actual', '0')}** |")
    md.append(f"| Forecasted Spend | **${budget.get('forecast', '0')}** |")
    md.append("")

    md.append("---")
    md.append(f"*Dashboard auto-generated by [`dashboard/generate_report.py`](generate_report.py) · Updated weekly via GitHub Actions*")
    md.append("")

    return "\n".join(md)


if __name__ == "__main__":
    output_path = os.getenv("OUTPUT_PATH", "dashboard/DASHBOARD.md")
    report = generate_markdown()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✅ Dashboard written to {output_path}")
