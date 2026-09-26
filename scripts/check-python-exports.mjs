// Python-export check. Every lab's download button produces a runnable Python
// script that must mirror the on-screen implementation. This script bundles each
// export module (labs/<area>/python.ts, labs/llm/ragPython.ts,
// components/rlPython.ts) with esbuild, calls every PYTHON_SAMPLES entry (see
// utils/pythonSamples.ts), and verifies each generated script:
//   • parses (ast),
//   • leaks no JavaScript literals (true/false/null/undefined/NaN/Infinity) or `${`,
//   • references no undefined names (scope-agnostic heuristic),
//   • and, with --run, executes cleanly when all of its imports are installed.
//
// Usage: node scripts/check-python-exports.mjs [--area <name> | --file <module.ts>] [--run] [--keep]
// Scratch files go under .claude/tmp/python-exports/ and are removed afterwards
// unless --keep is given.
import { build } from 'esbuild';
import { existsSync, mkdirSync, readdirSync, rmdirSync, statSync, unlinkSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2);
const opt = (name) => argv.includes(name);
const areaArg = argv.includes('--area') ? argv[argv.indexOf('--area') + 1] : null;
const RUN = opt('--run');
const KEEP = opt('--keep');

// ---- discover export modules -------------------------------------------------
const modules = [];
for (const area of readdirSync(join(root, 'labs')).sort()) {
  const file = join(root, 'labs', area, 'python.ts');
  if (existsSync(file)) modules.push({ area, file });
}
for (const [area, rel] of [['rag', 'labs/llm/ragPython.ts'], ['rl', 'components/rlPython.ts']]) {
  const file = join(root, rel);
  if (existsSync(file)) modules.push({ area, file });
}
const fileArg = argv.includes('--file') ? resolve(argv[argv.indexOf('--file') + 1]) : null;
const selected = fileArg
  ? [{ area: fileArg.split(sep).slice(-2).join('-').replace(/\.tsx?$/, '').replace(/[^A-Za-z0-9_-]/g, '_'), file: fileArg }]
  : areaArg ? modules.filter((m) => m.area === areaArg) : modules;
if (!selected.length) {
  console.error(`No export module for area "${areaArg}". Known areas: ${modules.map((m) => m.area).join(', ')}`);
  process.exit(2);
}

// ---- scratch space (only ever inside .claude/tmp/python-exports) -------------
const scratchRoot = join(root, '.claude', 'tmp', 'python-exports');
const insideScratch = (p) => resolve(p).startsWith(scratchRoot + sep);
function wipe(dir) {
  if (!existsSync(dir) || !insideScratch(dir)) return;
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) { wipe(p); rmdirSync(p); } else unlinkSync(p);
  }
}

// ---- the Python side of the check ----------------------------------------------
const CHECKER = String.raw`import ast, builtins, importlib.util, io, json, os, subprocess, sys, tokenize

JS_LITERALS = {"true", "false", "null", "undefined", "NaN", "Infinity"}

def defined_names(tree):
    names = set(dir(builtins)) | {"__name__", "__file__", "__doc__", "__builtins__"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
    return names

def check(job):
    path, run, timeout = job["path"], job["run"], job["timeout"]
    src = open(path, encoding="utf-8").read()
    problems = []
    if not src.strip():
        return {"problems": ["empty script"], "run": "not run"}
    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError as e:
        return {"problems": ["SyntaxError line %s: %s" % (e.lineno, e.msg)], "run": "not run"}
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.NAME and tok.string in JS_LITERALS:
            problems.append("JavaScript literal '%s' at line %d" % (tok.string, tok.start[0]))
    leak = "$" + "{"  # a leaked JavaScript template placeholder
    if leak in src:
        problems.append("leaked template placeholder at line %d" % (src[: src.index(leak)].count("\n") + 1))
    star = any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names) for n in ast.walk(tree))
    if not star:
        names = defined_names(tree)
        seen = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in names and node.id not in seen:
                seen.add(node.id)
                problems.append("undefined name '%s' at line %d" % (node.id, node.lineno))
    status = "not run"
    if run and not problems and not job.get("noRun"):
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                roots.add(node.module.split(".")[0])
        missing = sorted(r for r in roots if importlib.util.find_spec(r) is None)
        if missing:
            status = "skipped (missing " + ", ".join(missing) + ")"
        else:
            env = dict(os.environ, MPLBACKEND="Agg", PYTHONHASHSEED="0")
            try:
                p = subprocess.run([sys.executable, path], capture_output=True, text=True,
                                   timeout=timeout, cwd=os.path.dirname(path), env=env)
                if p.returncode == 0:
                    status = "ran ok"
                else:
                    lines = [l for l in p.stderr.strip().splitlines() if l.strip()] or ["(no stderr)"]
                    problems.append("runtime error: " + lines[-1])
                    status = "failed"
            except subprocess.TimeoutExpired:
                problems.append("timed out after %ss" % timeout)
                status = "timeout"
    elif job.get("noRun"):
        status = "not run (noRun)"
    return {"problems": problems, "run": status}

if __name__ == "__main__":
    jobs = json.load(open(sys.argv[1], encoding="utf-8"))
    print(json.dumps([dict(check(j), name=j["name"]) for j in jobs]))
`;

// ---- main ---------------------------------------------------------------------
let failures = 0;
let total = 0;
mkdirSync(scratchRoot, { recursive: true });
try {
  for (const { area, file } of selected) {
    const dir = join(scratchRoot, area);
    wipe(dir);
    mkdirSync(dir, { recursive: true });
    const header = `\n== ${area}  (${file.slice(root.length + 1)})`;
    let samples;
    try {
      const out = await build({
        entryPoints: [file], bundle: true, platform: 'node', format: 'esm', write: false,
        packages: 'external', logLevel: 'silent', jsx: 'automatic', target: 'node20',
        loader: { '.css': 'empty' },
      });
      const bundlePath = join(dir, 'module.mjs');
      writeFileSync(bundlePath, out.outputFiles[0].text);
      const mod = await import(`${pathToFileURL(bundlePath).href}?t=${Date.now()}`);
      samples = mod.PYTHON_SAMPLES;
    } catch (err) {
      console.log(`${header}\n  FAIL  could not load module: ${String(err?.message ?? err).split('\n')[0]}`);
      failures++;
      continue;
    }
    if (!Array.isArray(samples) || samples.length === 0) {
      console.log(`${header}\n  FAIL  no PYTHON_SAMPLES exported (see utils/pythonSamples.ts)`);
      failures++;
      continue;
    }
    const jobs = [];
    const names = new Set();
    const buildErrors = [];
    for (const s of samples) {
      const name = String(s.name);
      if (names.has(name) || !/^[A-Za-z0-9_.-]+$/.test(name)) { buildErrors.push(`bad or duplicate sample name "${name}"`); continue; }
      names.add(name);
      let code;
      try { code = s.code(); } catch (err) { buildErrors.push(`${name}: code() threw ${String(err?.message ?? err)}`); continue; }
      if (typeof code !== 'string') { buildErrors.push(`${name}: code() did not return a string`); continue; }
      const sampleDir = join(dir, name);
      mkdirSync(sampleDir, { recursive: true });
      const path = join(sampleDir, `${area}__${name}.py`);
      writeFileSync(path, code);
      jobs.push({ name, path, run: RUN, timeout: s.timeoutSec ?? 90, noRun: !!s.noRun });
    }
    const jobsPath = join(dir, 'jobs.json');
    const checkerPath = join(dir, 'check_exports.py');
    writeFileSync(jobsPath, JSON.stringify(jobs));
    writeFileSync(checkerPath, CHECKER);
    const res = spawnSync('python3', [checkerPath, jobsPath], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
    console.log(header);
    for (const e of buildErrors) { console.log(`  FAIL  ${e}`); failures++; }
    if (res.status !== 0) {
      console.log(`  FAIL  python checker crashed: ${(res.stderr || '').trim().split('\n').pop()}`);
      failures++;
      continue;
    }
    for (const r of JSON.parse(res.stdout)) {
      total++;
      if (r.problems.length) {
        failures++;
        console.log(`  FAIL  ${r.name}  [${r.run}]`);
        for (const p of r.problems.slice(0, 8)) console.log(`          - ${p}`);
      } else {
        console.log(`  ok    ${r.name}  [${r.run}]`);
      }
    }
  }
} finally {
  if (!KEEP) {
    for (const { area } of selected) {
      const dir = join(scratchRoot, area);
      wipe(dir);
      if (existsSync(dir)) rmdirSync(dir);
    }
    if (existsSync(scratchRoot) && readdirSync(scratchRoot).length === 0) rmdirSync(scratchRoot);
  }
}
console.log(`\n${total} sample(s) checked, ${failures} failure(s).${KEEP ? `  Scripts kept in ${scratchRoot}` : ''}`);
process.exit(failures ? 1 : 0);
