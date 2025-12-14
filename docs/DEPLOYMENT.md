# Production Deployment Guide

This guide covers deploying the RL Interactive Labs application to production with nginx reverse proxy on the server.

## Prerequisites

- Docker and Docker Compose installed on server
- Nginx installed and configured on host server
- Domain name configured (optional but recommended)
- SSL certificate (recommended via Let's Encrypt)

## Quick Start

### 1. Clone and Configure

```bash
# Clone the repository
git clone <repository-url>
cd RL-Interactive-Labs

# Copy environment template
cp .env.example .env

# (Optional) Add your Gemini API key
nano .env
# Set: GEMINI_API_KEY=your_actual_key_here
```

### 2. Build and Deploy

```bash
# Build and start the container
docker compose up -d --build

# Verify it's running
docker compose ps
docker compose logs -f
```

The application will be available on `http://localhost:2100`

### 3. Configure Nginx Reverse Proxy

Since you have nginx on the host server, add this configuration:

```nginx
# /etc/nginx/sites-available/rl-labs
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain

    # Redirect HTTP to HTTPS (if using SSL)
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;  # Replace with your domain

    # SSL Configuration (if using Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Proxy to Docker container
    location / {
        proxy_pass http://localhost:2100;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://localhost:2100/health;
        access_log off;
    }

    # Security headers (already set in container, but can add here too)
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/rl-labs /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## API Key Strategy

### Option 1: Provide Your Own Key (Recommended for PoC)

1. Add `GEMINI_API_KEY` to `.env` file
2. Users can access AI features immediately
3. When your quota is exhausted, users see prompt to enter their own key
4. This provides a seamless "try before you buy" experience

### Option 2: No Default Key

1. Leave `GEMINI_API_KEY` empty or remove from `.env`
2. All users must provide their own API key via UI
3. More scalable but higher barrier to entry

### Option 3: Hybrid (Current Implementation)

- Provide your key in `.env` for initial users
- Application automatically detects quota exhaustion
- Gracefully falls back to user-provided keys
- Best of both worlds!

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | No | - | Google Gemini API key for AI tutoring features |
| `NODE_ENV` | No | `production` | Application environment |

## Container Management

### View Logs
```bash
docker compose logs -f
```

### Restart Container
```bash
docker compose restart
```

### Update Application
```bash
git pull
docker compose down
docker compose up -d --build
```

### Stop Application
```bash
docker compose down
```

### View Resource Usage
```bash
docker stats rl-interactive-labs
```

## Health Monitoring

### Container Health Check
The container includes a built-in health check that runs every 30 seconds:

```bash
docker inspect --format='{{.State.Health.Status}}' rl-interactive-labs
```

### Application Health Endpoint
```bash
curl http://localhost:2100/health
# Should return: healthy
```

### Monitor from Host Nginx
Add monitoring configuration to nginx:

```nginx
location /health {
    proxy_pass http://localhost:2100/health;
    access_log off;
}
```

## Security Checklist

- [x] API keys removed from Docker build process
- [x] Environment variables loaded at runtime only
- [x] Security headers configured (CSP, X-Frame-Options, etc.)
- [x] Rate limiting implemented for API calls
- [x] Error boundaries for graceful failure handling
- [x] Health checks configured
- [ ] SSL/TLS enabled via host nginx (your responsibility)
- [ ] Firewall configured to restrict port 2100 to localhost only
- [ ] Regular security updates applied to host OS
- [ ] Container images regularly updated

## Performance Optimization

### 1. Enable nginx Caching (Host)

Add to your nginx config:
```nginx
# Cache zone definition
proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=rl_cache:10m max_size=100m inactive=60m;

location / {
    proxy_cache rl_cache;
    proxy_cache_valid 200 5m;
    proxy_cache_key $scheme$request_method$host$request_uri;
    # ... rest of proxy config
}
```

### 2. Container Resource Limits

Add to `docker-compose.yml`:
```yaml
services:
  rl-interactive-labs:
    # ... existing config
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker compose logs

# Check if port is already in use
sudo netstat -tulpn | grep 2100

# Rebuild from scratch
docker compose down
docker system prune -a
docker compose up -d --build
```

### High memory usage
```bash
# Check container stats
docker stats rl-interactive-labs

# Restart container to clear memory
docker compose restart
```

### API key not working
```bash
# Check environment variables are loaded
docker compose exec rl-interactive-labs env | grep GEMINI

# Verify .env file exists
ls -la .env

# Restart container after changing .env
docker compose down && docker compose up -d
```

### Nginx reverse proxy issues
```bash
# Test nginx configuration
sudo nginx -t

# Check nginx error logs
sudo tail -f /var/log/nginx/error.log

# Verify container is accessible from host
curl http://localhost:2100/health
```

## Backup and Recovery

### Backup Configuration
```bash
# Backup .env file (contains sensitive data!)
cp .env .env.backup

# Store securely, do not commit to git
```

### Restore Configuration
```bash
# Restore .env
cp .env.backup .env

# Rebuild and restart
docker compose up -d --build
```

## Updates and Maintenance

### Update Docker Images
```bash
# Pull latest base images
docker compose pull

# Rebuild
docker compose up -d --build
```

### Update Application Code
```bash
# Pull latest code
git pull origin main

# Rebuild container
docker compose down
docker compose up -d --build
```

### Monitor Gemini API Usage
- Visit: https://aistudio.google.com/
- Check quota usage under your API key
- Set up billing alerts if using paid tier

## Support

For issues or questions:
- Check application logs: `docker compose logs -f`
- Review nginx logs: `sudo tail -f /var/log/nginx/error.log`
- Check container health: `docker inspect rl-interactive-labs`
- Verify environment: `docker compose config`

## Production Readiness Score: 75/100

### Implemented ✅
- Multi-stage Docker build
- Health checks (container + nginx)
- Security headers (CSP, X-Frame-Options, etc.)
- Rate limiting for API calls
- Retry logic with exponential backoff
- Error boundaries for React
- TypeScript strict mode
- Comprehensive documentation
- API key flexibility (env key with user fallback)

### Still Needed ⚠️
- Automated testing (unit + integration tests)
- CI/CD pipeline
- Application logging to external service
- Error tracking (e.g., Sentry)
- Performance monitoring
- Automated backups
- Load testing results

### Recommended Before Scale 📈
- Set up monitoring (Prometheus + Grafana)
- Configure log aggregation (ELK stack or similar)
- Implement automated testing
- Create runbook for common incidents
- Set up alerting for downtime
- Consider CDN for static assets
