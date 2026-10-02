#!/bin/sh
# Container entrypoint: make sure the models are present and verified, then run the command.
set -eu

# ensure_model skips files that already exist and match their pinned SHA-256.
python /app/scripts/download_models.py

exec "$@"
