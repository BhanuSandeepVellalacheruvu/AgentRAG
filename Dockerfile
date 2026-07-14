FROM public.ecr.aws/lambda/python:3.12

# Copy the project files
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install dependencies and the project package
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache-dir .[api,agents,retrieval]

# Set CMD to the Mangum handler
CMD ["agentrag.api.main.handler"]
