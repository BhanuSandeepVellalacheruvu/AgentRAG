#!/usr/bin/env python3
"""CDK app entrypoint."""

import aws_cdk as cdk
from stack import AgentRagStack


def main() -> None:
    """Instantiate the CDK App and Stack."""
    app = cdk.App()
    AgentRagStack(app, "AgentRagStack")
    app.synth()


if __name__ == "__main__":
    main()
