#!/usr/bin/env python3
"""Application-owned Docker rollout hook, streamed over Tailscale SSH.

Only the selected services are rebuilt. Receipts and retained image tags live
outside the application checkout. Neither secrets nor container environments
are returned to the coordinator. A lost connection must be reconciled by status.
"""
import base64
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def command(args, cwd, capture=True):
    result = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        # Avoid copying build logs or configuration containing secrets into a receipt.
        raise RuntimeError(f'{args[0]} {args[1]} failed with exit {result.returncode}')
    return result.stdout.decode().strip()


def save(path, data):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def compose(config, override=None):
    args = ['docker', 'compose', '-p', config['compose_project']]
    for filename in config['compose_files']:
        args += ['-f', filename]
    if override:
        args += ['-f', str(override)]
    return args


def inspect(config):
    root = config['path']
    if command(['git', 'status', '--porcelain'], root):
        raise RuntimeError('VPS checkout is dirty')
    images, containers = {}, {}
    for service in config['services']:
        ids = command(compose(config) + ['ps', '-q', service], root).splitlines()
        if len(ids) != 1:
            raise RuntimeError(f'{service}: expected one running container')
        state = json.loads(command(['docker', 'inspect', '--format',
            '{{json .State}}', ids[0]], root))
        if state.get('Status') != 'running' or state.get('Health', {}).get('Status', 'healthy') != 'healthy':
            raise RuntimeError(f'{service}: container is not healthy')
        images[service] = command(['docker', 'inspect', '--format', '{{.Image}}', ids[0]], root)
        containers[service] = ids[0]
    state = {'head': command(['git', 'rev-parse', 'HEAD'], root), 'images': images, 'containers': containers}
    state['fingerprint'] = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
    return {'status': 'ready', **state}


def health(config, expected_images):
    deadline = time.monotonic() + config.get('health_timeout', 120)
    last = 'health checks did not pass'
    while time.monotonic() < deadline:
        try:
            current = inspect(config)
            if current['images'] != expected_images:
                raise RuntimeError('Running image does not match the built image')
            for url in config['health_urls']:
                with urllib.request.urlopen(url, timeout=10) as response:
                    if response.status != 200:
                        raise RuntimeError('HTTP health check failed')
            return current
        except Exception as exc:
            last = str(exc)
            time.sleep(2)
    raise RuntimeError(last)


def start(config, override):
    command(compose(config, override) + ['up', '-d', '--no-deps', '--no-build', '--pull', 'never',
            '--force-recreate', '--wait', '--wait-timeout', str(config.get('health_timeout', 120)),
            *config['services']], config['path'])


def restore(config, state, override):
    root = config['path']
    if command(['git', 'status', '--porcelain'], root):
        raise RuntimeError('Refusing rollback over a dirty checkout')
    for image in state['images'].values():
        command(['docker', 'image', 'inspect', '--format', '{{.Id}}', image], root)
    command(['git', 'checkout', '--detach', state['head']], root)
    save(override, {'services': {s: {'image': i} for s, i in state['images'].items()}})
    start(config, override)
    return health(config, state['images'])


def execute(operation, config, request, state_dir):
    root = config['path']
    if operation == 'doctor':
        return inspect(config)
    request_id = request['request_id']
    if not re.fullmatch(r'[a-z0-9-]{1,100}', request_id):
        raise ValueError('Invalid request ID')
    receipt_path = state_dir / (request_id + '.json')
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        if previous['catalog_sha256'] != request['catalog_sha256'] or previous['candidate_sha'] != request['candidate_sha']:
            raise RuntimeError('Request identity reused for different content')
        return previous
    if operation == 'status':
        raise RuntimeError('No saved receipt: deployment may not have started; investigate before retrying')
    prior = inspect(config)
    receipt = {k: request[k] for k in ('application', 'request_id', 'catalog_sha256', 'candidate_sha')}
    receipt.update(status='deploying', operation=operation, previous=prior)
    override = state_dir / (request_id + '-compose.json')
    if operation == 'rollback':
        rollback = request['rollback']
        original_path = state_dir / (rollback['original_request_id'] + '.json')
        original = json.loads(original_path.read_text())
        if original.get('status') != 'deployed' or prior['head'] != rollback['expected_current']:
            raise RuntimeError('Rollback no longer matches the live release')
        if prior['images'] != original['deployed_images'] or original['previous']['head'] != rollback['target']:
            raise RuntimeError('Rollback image identity mismatch')
        save(receipt_path, receipt)
        try:
            restore(config, original['previous'], override)
            receipt.update(status='rolled_back', deployment_revision=rollback['target'])
        except Exception as exc:
            receipt.update(status='failed', error=str(exc))
        save(receipt_path, receipt)
        return receipt
    if operation != 'update':
        raise RuntimeError('Unknown operation')
    if prior['fingerprint'] != request['baseline']['fingerprint']:
        raise RuntimeError('Live release changed since preparation')
    candidate = request['candidate_sha']
    if not re.fullmatch(r'[a-f0-9]{40}', candidate):
        raise ValueError('Invalid commit')
    command(['git', 'fetch', 'origin', candidate], root)
    command(['git', 'merge-base', '--is-ancestor', prior['head'], candidate], root)
    # Do not rebuild an already tested/current release, but still verify health.
    current_path = state_dir / 'current.json'
    known_current = json.loads(current_path.read_text()) if current_path.exists() else {}
    if prior['head'] == candidate and known_current.get('deployment_revision') == candidate and known_current.get('deployed_images') == prior['images']:
        health(config, prior['images'])
        receipt.update(status='unchanged', deployment_revision=candidate)
        save(receipt_path, receipt)
        return receipt
    # Protect previous images from ordinary dangling-image cleanup.
    suffix = hashlib.sha256(request_id.encode()).hexdigest()[:24]
    for service, image in prior['images'].items():
        command(['docker', 'image', 'tag', image, f'model-update/{request["application"]}:{suffix}-previous-{service}'], root)
    save(receipt_path, receipt)
    try:
        command(['git', 'merge', '--ff-only', candidate], root)
        tags = {s: f'model-update/{request["application"]}:{suffix}-{s}' for s in config['services']}
        save(override, {'services': {s: {'image': tag} for s, tag in tags.items()}})
        command(compose(config, override) + ['build', *config['services']], root)
        receipt['deployed_images'] = {s: command(['docker', 'image', 'inspect', '--format', '{{.Id}}', tag], root)
                                      for s, tag in tags.items()}
        save(receipt_path, receipt)
        start(config, override)
        health(config, receipt['deployed_images'])
        receipt.update(status='deployed', deployment_revision=candidate, rollback_target=prior['head'])
        save(current_path, receipt)
    except Exception as exc:
        receipt.update(status='failed', error=str(exc))
        try:
            restore(config, prior, override)
            receipt['recovery'] = 'Previous source and images restored and healthy'
        except Exception as recovery:
            receipt['recovery'] = 'Automatic recovery failed: ' + str(recovery)
    save(receipt_path, receipt)
    return receipt


def main():
    operation, encoded = sys.argv[1:]
    payload = json.loads(base64.urlsafe_b64decode(encoded))
    config, request = payload['deployment'], payload['request']
    app = request['application']
    if not re.fullmatch(r'[a-z0-9-]{1,96}', app):
        raise ValueError('Invalid application')
    if not Path(config['path']).is_absolute():
        raise ValueError('Deployment root must be absolute')
    # Doctor is read-only: no directory or lock creation on production.
    if operation == 'doctor':
        baseline = inspect(config)
        if request.get('branch'):
            remote = command(['git', 'ls-remote', '--exit-code', 'origin', 'refs/heads/' + request['branch']], config['path'])
            baseline['source_head'] = remote.split()[0]
        print(json.dumps(baseline))
        return
    state_dir = Path.home() / '.local/state/model-updates' / app
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (state_dir / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(execute(operation, config, request, state_dir)))


if __name__ == '__main__':
    main()
