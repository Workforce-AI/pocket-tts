# Deployment Guide — AWS EC2

Straightforward deployment using Docker on a single EC2 instance.

---

## Prerequisites

- AWS account with EC2 access
- Your fork cloned: `github.com/Workforce-AI/pocket-tts`
- A HuggingFace account with access to `kyutai/pocket-tts` (required for voice cloning)
- A strong API key ready (generate one: `openssl rand -hex 32`)

---

## Step 1 — Launch EC2 Instance

1. Go to **EC2 → Launch Instance**
2. Set:
   - **Name:** `pocket-tts`
   - **AMI:** Ubuntu 24.04 LTS
   - **Instance type:** `t3.large` (2 vCPU, 8GB RAM)
   - **Storage:** 20GB gp3 (default is fine)
   - **Security group:** create new, add these inbound rules:

| Type | Port | Source |
|---|---|---|
| SSH | 22 | Your IP only |
| Custom TCP | 8000 | Your app servers' IPs (or `0.0.0.0/0` temporarily for testing) |

3. Create or select a key pair — save the `.pem` file
4. Launch

---

## Step 2 — Connect to the Instance

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

---

## Step 3 — Install Docker

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
sudo usermod -aG docker ubuntu
newgrp docker
```

Verify:
```bash
docker --version
```

---

## Step 4 — Clone Your Fork

```bash
git clone https://github.com/Workforce-AI/pocket-tts.git
cd pocket-tts
```

---

## Step 5 — Set Environment Variables

```bash
# Create env file (never commit this)
cat > .env << 'EOF'
POCKET_TTS_API_KEY=your-secret-api-key-here
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
EOF
```

Get your HuggingFace token from: `huggingface.co → Settings → Access Tokens`
Make sure the token has **read** access and your account has accepted the terms at `huggingface.co/kyutai/pocket-tts`.

---

## Step 6 — Update docker-compose.yaml

The default `docker-compose.yaml` doesn't pass env vars or enable quantization. Edit it:

```bash
nano docker-compose.yaml
```

Replace the contents with:

```yaml
services:
  pocket-tts:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - pocket_tts_cache:/root/.cache/pocket_tts
      - hf_cache:/root/.cache/huggingface
    command: ["serve", "--host", "0.0.0.0", "--port", "8000", "--quantize"]
    restart: unless-stopped

volumes:
  pocket_tts_cache:
  hf_cache:
```

---

## Step 7 — Build and Start

```bash
docker compose up -d --build
```

First build takes 3–5 minutes (installs dependencies). Model weights download on first request (~500MB, takes 1–2 minutes).

Check it's running:
```bash
docker compose logs -f
```

You should see `Uvicorn running on http://0.0.0.0:8000`. Hit `Ctrl+C` to stop following logs — the container keeps running.

---

## Step 8 — Verify

```bash
# Health check (no auth required)
curl http://<EC2_PUBLIC_IP>:8000/health

# Expected: {"status":"healthy"}
```

```bash
# List voices (auth required)
curl http://<EC2_PUBLIC_IP>:8000/voices \
  -H "X-API-Key: your-secret-api-key-here"
```

```bash
# Quick TTS test
curl -X POST http://<EC2_PUBLIC_IP>:8000/tts \
  -H "X-API-Key: your-secret-api-key-here" \
  -F "text=Hello, this is a test." \
  -F "voice_url=alba" \
  --output test.wav

# Play it
open test.wav   # macOS
# or download and play locally
```

---

## Step 9 — Pre-cache Your Avatar Voices

Do this once per avatar voice. Voice cache resets on container restart, so run these after every deploy.

```bash
# From a WAV file
curl -X POST http://<EC2_PUBLIC_IP>:8000/voices/cache \
  -H "X-API-Key: your-secret-api-key-here" \
  -F "name=avatar-sarah" \
  -F "voice_wav=@/path/to/sarah_voice.wav"

# Save the returned voice_id in your app config
# {"voice_id": "f47ac10b-...", "name": "avatar-sarah"}
```

---

## Step 10 — Keep It Running After Reboots

The `restart: unless-stopped` in docker-compose handles container crashes. To survive EC2 reboots, enable Docker autostart:

```bash
sudo systemctl enable docker
```

The container restarts automatically because of `restart: unless-stopped`.

---

## Updating After Code Changes

```bash
cd pocket-tts
git pull origin main
docker compose up -d --build
```

Old container is replaced, new one starts. Downtime is ~30 seconds during build.

---

## Useful Commands

```bash
# View live logs
docker compose logs -f

# Stop the server
docker compose down

# Restart without rebuilding
docker compose restart

# Check resource usage
docker stats
```

---

## Swagger UI

Once running, full API docs are at:
```
http://<EC2_PUBLIC_IP>:8000/docs
```

---

## Next: Add HTTPS (Strongly Recommended)

The API key travels in plain text over HTTP. Before going to production:

1. Register a domain and point an A record to your EC2 IP
2. Create an **AWS ALB** (Application Load Balancer)
3. Request a free cert via **AWS Certificate Manager**
4. ALB listens on 443 (HTTPS), forwards to EC2 port 8000 (HTTP)

This is a separate setup. Until then, restrict port 8000 in your security group to only your app servers' IPs rather than `0.0.0.0/0`.
