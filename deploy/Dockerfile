FROM rust:1.76-bookworm AS builder

ARG FEATURES="live"
ARG PROFILE="release"

WORKDIR /opt/paraphina
COPY . .

RUN cargo build -p paraphina --bin paraphina_live --features "${FEATURES}" --profile "${PROFILE}"

FROM debian:bookworm-slim
RUN apt-get update -y && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

RUN useradd -m -r -s /bin/bash paraphina
WORKDIR /opt/paraphina

COPY --from=builder /opt/paraphina/target/release/paraphina_live /opt/paraphina/paraphina_live

ENV PARAPHINA_LIVE_OUT_DIR=/var/lib/paraphina/out
ENV PARAPHINA_TELEMETRY_MODE=jsonl

RUN mkdir -p /var/lib/paraphina/out && chown -R paraphina:paraphina /var/lib/paraphina
USER paraphina

EXPOSE 9898
ENTRYPOINT ["/opt/paraphina/paraphina_live"]
