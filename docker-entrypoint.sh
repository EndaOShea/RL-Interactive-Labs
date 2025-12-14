#!/bin/sh
set -e

# Inject runtime environment variables into config.js
CONFIG_FILE="/usr/share/nginx/html/config.js"

if [ -f "$CONFIG_FILE" ]; then
  echo "Injecting runtime configuration..."

  # Replace placeholder with actual API key (or empty string if not set)
  sed -i "s|__GEMINI_API_KEY__|${GEMINI_API_KEY:-}|g" "$CONFIG_FILE"

  echo "Configuration injected successfully"
else
  echo "Warning: config.js not found at $CONFIG_FILE"
fi

# Start nginx
exec "$@"
