# Model catalogue updates

The private `update_projects` coordinator runs this application's
`.model-update.json` commands locally, using one saved catalogue for the fleet.
Run the offline adapter with:

```sh
python3 scripts/sync-model-catalog.py --catalog /absolute/path/catalog.json
python3 scripts/sync-model-catalog.py --catalog /absolute/path/catalog.json --check
```

`model-catalog.policy.json` owns the approved model IDs and default choices.
The adapter refreshes their prices, limits and calling metadata. New IDs require
an application policy review and payload tests; missing/deprecated models or
incompatible parameters stop the update. A disappearing ID is not proof of a
replacement. Never regenerate by fetching a moving URL in this application.

Validation commands in `.model-update.json` run before committing or pushing.
Git pushes do not trigger CI or deployment. The coordinator then sends the
tested app-owned `scripts/deploy-model-update.py` hook over Tailscale SSH: it
checks the recorded baseline, fetches the exact commit, builds/recreates only
the listed services, checks image identity and health, and records a receipt.
An update failure restores the previous Git revision and retained Docker image
IDs. Receipts live on the VPS in `~/.local/state/model-updates/<app>/`.
Do not prune the retained `model-update/` images before the rollback window ends.
The hook does not run database migrations or change volumes, API keys or CSP.

The initial integration uses explicit replacements for absent legacy Gemini
and DeepSeek IDs. Gemini Flash Lite remains the economical browser default;
server-funded policies include price ceilings. Embeddings and local Ollama
models are outside this text catalogue. API payload checks are offline and do
not claim live paid-provider verification.
