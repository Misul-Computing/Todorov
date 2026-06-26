"""Push a self-contained Kaggle script that inlines the quant/ package + a runner.

The Kaggle kernel runtime accepts a single script.py, so we concatenate the
package modules (rotate, codebook, quantize, eval) and a runner snippet into one
file, push it via kaggle_runtime.run_on_kaggle_with_artifacts, and save the
output files (runs/*/summary.json) locally.

Auth: reads KAGGLE_API_TOKEN from .mcp.json (KGAT_* token), falling back to
~/.kaggle/kaggle.json.

Usage:
  python kaggle_push.py <runner_module> [--timeout 900] [--out runs/<name>]

Examples:
  python kaggle_push.py run_quant --out runs/pkg_repro
  python kaggle_push.py run_picker --out runs/picker
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "quant"


def load_token():
    """Read KAGGLE_API_TOKEN from .mcp.json, then ~/.kaggle/kaggle.json."""
    mcp = ROOT / ".mcp.json"
    if mcp.exists():
        cfg = json.loads(mcp.read_text())
        token = (cfg.get("mcpServers", {}).get("kaggle-exec", {})
                 .get("env", {}).get("KAGGLE_API_TOKEN", ""))
        if token:
            os.environ["KAGGLE_API_TOKEN"] = token
            return token
    kj = Path.home() / ".kaggle" / "kaggle.json"
    if kj.exists():
        d = json.loads(kj.read_text())
        if d.get("api_token"):
            os.environ["KAGGLE_API_TOKEN"] = d["api_token"]
            return d["api_token"]
    return None


def inline_package():
    """Concatenate quant/{rotate,codebook,quantize,eval}.py into one namespace.

    Base64-encodes each module and registers it in sys.modules under its bare
    name before exec'ing the next, so `from rotate import ...` resolves to the
    already-exec'd module.  Base64 avoids all string-escaping pitfalls.
    """
    import base64
    order = ["rotate", "codebook", "vq", "quantize", "eval", "picker", "pack", "reasoning", "hablc", "aqlm", "finetune", "dbf", "adaptive", "e8lattice", "e8p", "ldlq"]
    parts = [
        "# -*- auto-inlined quant/ package for Kaggle -*-\n",
        "import sys, types, base64\n",
        "def _reg(name, b64):\n",
        "    m = types.ModuleType(name)\n",
        "    exec(compile(base64.b64decode(b64).decode(), name, 'exec'), m.__dict__)\n",
        "    sys.modules[name] = m\n",
        "    return m\n",
    ]
    for mod in order:
        src = (PKG / f"{mod}.py").read_text()
        src = src.replace("from .rotate import", "from rotate import")
        src = src.replace("from .codebook import", "from codebook import")
        src = src.replace("from .vq import", "from vq import")
        src = src.replace("from .quantize import", "from quantize import")
        src = src.replace("from .eval import", "from eval import")
        src = src.replace("from .picker import", "from picker import")
        src = src.replace("from .pack import", "from pack import")
        src = src.replace("from .reasoning import", "from reasoning import")
        src = src.replace("from .hablc import", "from hablc import")
        src = src.replace("from .aqlm import", "from aqlm import")
        src = src.replace("from .finetune import", "from finetune import")
        src = src.replace("from .dbf import", "from dbf import")
        src = src.replace("from . import", "from quant import")
        # Also handle absolute imports (quant.X -> X)
        src = src.replace("from quant.rotate import", "from rotate import")
        src = src.replace("from quant.codebook import", "from codebook import")
        src = src.replace("from quant.vq import", "from vq import")
        src = src.replace("from quant.quantize import", "from quantize import")
        src = src.replace("from quant.eval import", "from eval import")
        src = src.replace("from quant.e8lattice import", "from e8lattice import")
        src = src.replace("from quant.e8p import", "from e8p import")
        src = src.replace("from quant.ldlq import", "from ldlq import")
        b64 = base64.b64encode(src.encode()).decode()
        parts.append(f'_reg("{mod}", "{b64}")\n')
    return "".join(parts)


def build_script(runner_path):
    """Inline the package + the runner's body into one script.

    Strips `if __name__ == "__main__":` guards (the inlined script always runs
    at top level) and calls main() once if the runner defines it; flat scripts
    (no main) just run their top-level code.
    """
    pkg_blob = inline_package()
    runner = Path(runner_path).read_text()
    runner = runner.replace('from quant.quantize import', 'from quantize import')
    runner = runner.replace('from quant.eval import', 'from eval import')
    runner = runner.replace('from quant.picker import', 'from picker import')
    runner = runner.replace('from quant.pack import', 'from pack import')
    runner = runner.replace('from quant.reasoning import', 'from reasoning import')
    runner = runner.replace('from quant.hablc import', 'from hablc import')
    runner = runner.replace('from quant.aqlm import', 'from aqlm import')
    runner = runner.replace('from quant.finetune import', 'from finetune import')
    runner = runner.replace('from quant.dbf import', 'from dbf import')
    runner = runner.replace('from quant.codebook import', 'from codebook import')
    runner = runner.replace('from quant.rotate import', 'from rotate import')
    runner = runner.replace('from quant.vq import', 'from vq import')
    runner = runner.replace('from quant.adaptive import', 'from adaptive import')
    runner = runner.replace('from quant.e8lattice import', 'from e8lattice import')
    runner = runner.replace('from quant.e8p import', 'from e8p import')
    runner = runner.replace('from quant.ldlq import', 'from ldlq import')
    lines = []
    for ln in runner.splitlines():
        if 'sys.path.insert' in ln and 'quant' in ln:
            continue
        if ln.strip().startswith('if __name__') or (ln.strip() in ('main()', '    main()')):
            continue
        lines.append(ln)
    runner = "\n".join(lines)
    has_main = 'def main(' in runner
    tail = "\n\nmain()\n" if has_main else "\n"
    return pkg_blob + "\n\n# ===== runner =====\n" + runner + tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runner", help="runner module path, e.g. run_quant.py")
    ap.add_argument("--out", default=None, help="local output dir for artifacts")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--internet", action="store_true", default=True)
    ap.add_argument("--save-script", default=None, help="write the inlined script here")
    args = ap.parse_args()

    token = load_token()
    if not token:
        print("ERROR: no KAGGLE_API_TOKEN found in .mcp.json or ~/.kaggle/kaggle.json",
              file=sys.stderr)
        sys.exit(1)
    print(f"[push] token loaded (KGAT_...{token[-6:]})", flush=True)

    runner_path = ROOT / args.runner if not os.path.isabs(args.runner) else Path(args.runner)
    if not runner_path.exists():
        print(f"ERROR: runner not found: {runner_path}", file=sys.stderr)
        sys.exit(1)

    script = build_script(runner_path)
    if args.save_script:
        Path(args.save_script).write_text(script)
        print(f"[push] inlined script saved to {args.save_script} "
              f"({len(script)} bytes)", flush=True)

    out_dir = args.out or f"runs/kaggle_{time.strftime('%Y%m%d_%H%M%S')}"
    out_path = ROOT / out_dir
    out_path.mkdir(parents=True, exist_ok=True)

    # Import the kaggle runtime engine
    sys.path.insert(0, os.path.join(os.environ.get("APPDATA", ""),
                                    "Python", "Python314", "site-packages"))
    try:
        from mcp_server_kaggle_exec.kaggle_runtime import run_on_kaggle_with_artifacts
    except ImportError as e:
        print(f"ERROR: mcp_server_kaggle_exec not importable: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[push] pushing to Kaggle T4 (timeout={args.timeout}s, internet={args.internet})", flush=True)
    t0 = time.time()
    result = run_on_kaggle_with_artifacts(
        code=script,
        local_output_dir=str(out_path),
        enable_gpu=True,
        enable_internet=args.internet,
        timeout=args.timeout,
        cleanup=True,
    )
    print(f"\n[push] done in {time.time() - t0:.1f}s  status={result['status']}", flush=True)
    print(f"[push] kernel_id={result.get('kernel_id')}  exec={result.get('execution_time')}s",
          flush=True)
    print("\n--- stdout (last 4000 chars) ---")
    print(result["stdout"][-4000:])
    if result["stderr"].strip():
        print("\n--- stderr (last 2000 chars) ---")
        print(result["stderr"][-2000:])
    files = result.get("output_files", [])
    if files:
        print(f"\n[push] {len(files)} output files saved to {out_path}")
        for f in files:
            print(f"  {f}")


if __name__ == "__main__":
    main()
