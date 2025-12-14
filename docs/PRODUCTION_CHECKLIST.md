# Production Deployment Checklist

Use this checklist before deploying to production.

## Pre-Deployment

### Code Quality
- [x] TypeScript strict mode enabled
- [x] No console errors in production build
- [x] Error boundaries implemented
- [ ] Unit tests written and passing
- [ ] Integration tests written and passing
- [ ] Code reviewed by team member

### Security
- [x] API keys removed from Dockerfile build args
- [x] Environment variables loaded at runtime only
- [x] Security headers configured (CSP, X-Frame-Options, etc.)
- [x] Rate limiting implemented for API calls
- [ ] SSL/TLS certificate installed (handled by nginx on host)
- [ ] Firewall configured to restrict Docker port to localhost
- [ ] .env file not committed to version control
- [ ] Secrets management solution in place (if needed)
- [ ] Security audit performed
- [ ] Dependencies scanned for vulnerabilities (`npm audit`)

### Performance
- [x] Production build optimized (`npm run build`)
- [x] Static assets have cache headers (1 year)
- [x] Gzip compression enabled
- [ ] Load testing performed
- [ ] Performance benchmarks documented
- [ ] CDN configured (if needed)
- [ ] Database indexes optimized (N/A - no database)

### Monitoring & Logging
- [x] Health check endpoint configured (`/health`)
- [x] Container health checks enabled
- [ ] Error tracking service configured (e.g., Sentry)
- [ ] Application logging to external service
- [ ] Uptime monitoring configured
- [ ] Alert rules defined
- [ ] Log aggregation configured

## Deployment Steps

### 1. Environment Setup
- [ ] Server provisioned with Docker installed
- [ ] Nginx installed and configured on host
- [ ] Domain name configured and DNS updated
- [ ] SSL certificate obtained (Let's Encrypt recommended)
- [ ] Firewall rules configured

### 2. Application Configuration
- [ ] Repository cloned to server
- [ ] `.env` file created from `.env.example`
- [ ] `GEMINI_API_KEY` added to `.env` (optional)
- [ ] File permissions set correctly (`chmod 600 .env`)

### 3. Docker Deployment
- [ ] Docker image built: `docker compose build`
- [ ] Container started: `docker compose up -d`
- [ ] Container status verified: `docker compose ps`
- [ ] Logs checked for errors: `docker compose logs -f`
- [ ] Health check passing: `curl http://localhost:2100/health`

### 4. Nginx Configuration
- [ ] Nginx site configuration created (see DEPLOYMENT.md)
- [ ] Configuration tested: `sudo nginx -t`
- [ ] Site enabled: `ln -s /etc/nginx/sites-available/rl-labs /etc/nginx/sites-enabled/`
- [ ] Nginx reloaded: `sudo systemctl reload nginx`
- [ ] HTTPS working correctly
- [ ] HTTP redirects to HTTPS

### 5. Verification
- [ ] Application accessible via domain
- [ ] All interactive labs working
- [ ] AI tutor responding (if API key provided)
- [ ] Error boundaries trigger correctly (test with intentional error)
- [ ] Rate limiting works (make 15+ requests in 1 minute)
- [ ] Health endpoint returns 200: `curl https://your-domain.com/health`
- [ ] Security headers present: `curl -I https://your-domain.com`

## Post-Deployment

### Immediate (Day 1)
- [ ] Monitor logs for errors: `docker compose logs -f`
- [ ] Check resource usage: `docker stats rl-interactive-labs`
- [ ] Verify API key quota usage (if using env key)
- [ ] Test all major features manually
- [ ] Notify users of deployment

### Short-term (Week 1)
- [ ] Monitor error rates
- [ ] Check performance metrics
- [ ] Review user feedback
- [ ] Optimize based on real-world usage
- [ ] Document any issues encountered

### Medium-term (Month 1)
- [ ] Review Gemini API costs and usage patterns
- [ ] Analyze user engagement metrics
- [ ] Plan feature improvements
- [ ] Review security audit findings
- [ ] Update dependencies: `npm update`

## Rollback Plan

If deployment fails, use this rollback procedure:

1. **Stop the container**
   ```bash
   docker compose down
   ```

2. **Check logs for errors**
   ```bash
   docker compose logs > deployment-error.log
   ```

3. **Restore previous version**
   ```bash
   git checkout <previous-commit>
   docker compose up -d --build
   ```

4. **Verify rollback**
   ```bash
   curl http://localhost:2100/health
   ```

5. **Investigate and fix**
   - Review error logs
   - Fix issues in development
   - Test thoroughly
   - Re-deploy

## Emergency Contacts

Document who to contact in case of issues:

- **DevOps Lead**: _______________
- **Backend Developer**: _______________
- **Security Team**: _______________
- **Hosting Provider Support**: _______________

## Maintenance Schedule

### Daily
- Check error logs
- Monitor API quota usage
- Verify health checks passing

### Weekly
- Review performance metrics
- Check for security updates
- Backup .env file (securely)

### Monthly
- Update dependencies
- Review and rotate API keys (if needed)
- Security audit
- Performance optimization review

## Known Limitations

Document current limitations:

1. **No Automated Tests**: Manual testing required before deployment
2. **Client-side API Keys**: API keys accessible in browser (by design for flexibility)
3. **Rate Limiting**: Conservative 12 RPM limit may need tuning based on usage
4. **No Database**: All state is client-side (intentional for this app)
5. **Single Container**: No horizontal scaling yet (not needed for current usage)

## Success Metrics

Define what "successful deployment" means:

- [ ] Uptime > 99.9%
- [ ] Page load time < 2 seconds
- [ ] Error rate < 0.1%
- [ ] AI tutor response time < 5 seconds
- [ ] User satisfaction score > 4/5

## Notes

Additional deployment notes:

- API Key Strategy: Using hybrid approach (env key + user fallback)
- Nginx reverse proxy handles SSL termination
- Docker container serves on port 80 internally, exposed as 2100 to host
- No database required - fully client-side application
- Rate limiting set to 12 RPM (conservative for Gemini free tier 15 RPM)

---

**Deployment Date**: _______________
**Deployed By**: _______________
**Version**: _______________
**Git Commit**: _______________
