# Model catalogue updates

The private `update_projects` coordinator runs this application's
`.model-update.json` commands locally, using one saved catalogue for the fleet.
Run the offline adapter with:

```sh
python3 scripts/sync-model-catalog.py --catalog /absolute/path/catalog.json
python3 scripts/sync-model-catalog.py --catalog /absolute/path/catalog.json --check
```

`model-catalog.policy.json` defines compatible model families and reviewed default
choices. Every active text model matching a family is discovered in the saved
catalogue; IDs outside these request profiles are recorded with exclusion reasons.
Prices, reasoning modes, parameter names and required effort levels are validated.
A missing/deprecated default or an incompatible new release stops the update for
review. Specialized APIs (audio, realtime, search, pro/Codex and agent endpoints)
are excluded unless this application implements their request protocol.
Never regenerate by fetching a moving URL in this application.

Validation commands in `.model-update.json` run before committing or pushing.
Git pushes do not trigger CI or deployment. The coordinator then sends the
tested app-owned `scripts/deploy-model-update.py` hook over Tailscale SSH: it
checks the recorded baseline, fetches the exact commit, builds/recreates only
the listed services, checks image identity and health, and records a receipt.
An update failure restores the previous Git revision and retained Docker image
IDs. Receipts live on the VPS in `~/.local/state/model-updates/<app>/`.
Do not prune the retained `model-update/` images before the rollback window ends.
The hook does not run database migrations or change volumes, API keys or CSP.

The September 2026 catalogue exposes 38 models (19 OpenAI, 10 Anthropic,
7 Google, 2 DeepSeek), including GPT-6 Astra/Sol/Luna and Claude Opus 5.5 /
Fable 5.1. Existing economical defaults remain. Payload tests exercise all models
and render/select the actual API-key panel. The HTML shell revalidates on reload;
hashed assets remain immutable. Keys remain in memory only.

API payload tests are offline; they do not claim live paid-provider verification.
