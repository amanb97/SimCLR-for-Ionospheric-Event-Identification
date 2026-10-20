FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg
COPY requirements.txt pyproject.toml ./
RUN python -m pip install --no-cache-dir 'torch>=2.2,<3' --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY scripts ./scripts
RUN python -m pip install --no-cache-dir --no-deps .
CMD ["python", "scripts/run_workflow.py", "--data-dir", "/data", "--out-dir", "/outputs/workflow", "--device", "cpu"]
