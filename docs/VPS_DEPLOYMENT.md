# VPS Deployment (Roadmap‑B)

## GitHub Merge Flow

1. Create a branch from `main`.
2. Open a PR and wait for required checks to pass.
3. Ensure PR is reviewed and approved.
4. Merge via GitHub UI (squash or merge commit; no force pushes).

Required checks:
- `cargo test --all`
- Live connector matrix tests (see CI workflow).
- Python test suite.

## VPS Provisioning Checklist

- OS: Ubuntu 22.04+ (or Debian Bookworm).
- Firewall: allow SSH, restrict `/metrics` to trusted IPs.
- Users: create `paraphina` system user.
- Disks: attach volume for `/var/lib/paraphina`.
- Time sync: enable `systemd-timesyncd` or `chrony`.
- Limits: set `nofile`/`nproc` per `deploy/bootstrap.sh`.

## Deploy (Git Clone / Pull)

```
sudo mkdir -p /opt/paraphina
sudo chown -R $USER:$USER /opt/paraphina
git clone https://github.com/<org>/paraphina /opt/paraphina
cd /opt/paraphina
cargo build -p paraphina --bin paraphina_live --features live,live_hyperliquid --release
```

Install systemd unit + env file:

```
sudo cp deploy/systemd/paraphina_live.service.template /etc/systemd/system/paraphina_live.service
sudo cp deploy/env/roadmap_b_shadow.env.example /etc/paraphina/paraphina_live.env
sudo systemctl daemon-reload
sudo systemctl enable paraphina_live
sudo systemctl start paraphina_live
```

Logs and output:
- Logs: `/var/log/paraphina/paraphina_live.log`
- Output: `/var/lib/paraphina/out`

## Deploy (Container)

```
docker build -f deploy/Dockerfile -t paraphina-live:roadmap-b .
docker compose -f deploy/docker-compose.yml up -d
```

## Run a Shadow Session

```
PARAPHINA_TRADE_MODE=shadow \
PARAPHINA_LIVE_CONNECTOR=hyperliquid \
PARAPHINA_LIVE_OUT_DIR=/var/lib/paraphina/out \
PARAPHINA_TELEMETRY_MODE=jsonl \
paraphina_live
```

## Tailing Logs

```
tail -f /var/log/paraphina/paraphina_live.log
tail -f /var/log/paraphina/paraphina_live.err
```
