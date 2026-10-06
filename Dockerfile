FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md requirements-reproduce.txt ./
COPY mcpricing ./mcpricing
RUN pip install --no-cache-dir -r requirements-reproduce.txt && pip install --no-cache-dir --no-deps .
ENTRYPOINT ["python", "-m", "mcpricing"]
CMD ["price"]
