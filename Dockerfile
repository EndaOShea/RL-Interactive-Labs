# Multi-stage build for production deployment

# Stage 1: Build the application
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Build the application
# Note: API keys should ONLY be provided at runtime, not build time
RUN npm run build

# Stage 2: Serve with nginx
FROM nginx:alpine

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom nginx configuration + the shared security-headers include
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY security-headers.conf /etc/nginx/security-headers.conf

# Expose port 2100
EXPOSE 2100

# Health check
# 127.0.0.1 (not localhost): nginx binds IPv4 only, but localhost resolves to
# ::1 first inside the container, which would fail the probe.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --quiet --tries=1 --spider http://127.0.0.1:2100/health || exit 1

# The official nginx image already provides an entrypoint that launches the
# server with this CMD — no custom wrapper needed.
CMD ["nginx", "-g", "daemon off;"]
