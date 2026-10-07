# wardcat serve as a container. The service listens on 0.0.0.0, so it refuses
# to start without an API key: pass WARDCAT_API_KEY (16+ characters).
#
#   docker build -t wardcat .
#   docker run --rm -p 8787:8787 -e WARDCAT_API_KEY \
#     -v "$PWD/policy.yaml:/etc/wardcat/policy.yaml:ro" wardcat
#
# The image carries the regex layer only. For NER, build FROM this image and
# install wardcat[ner] plus the SpaCy model you need.

# python:3.12-slim, pinned by digest so a rebuild gets the same base.
FROM python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels . \
 && pip wheel --no-cache-dir --wheel-dir /wheels "$(ls /wheels/wardcat-*.whl)[serve]"

FROM python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f
RUN useradd --create-home --uid 10001 wardcat
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links /wheels "wardcat[serve]" \
 && rm -rf /wheels
USER wardcat
ENV PYTHONUNBUFFERED=1
EXPOSE 8787
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8787/healthz', timeout=2).status == 200 else 1)"
ENTRYPOINT ["wardcat", "serve", "--host", "0.0.0.0", "--port", "8787"]
CMD ["--config", "/etc/wardcat/policy.yaml"]
