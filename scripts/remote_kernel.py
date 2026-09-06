#!/usr/bin/env python3
"""Drive a remote Jupyter kernel (e.g. a Kaggle session) over its REST/WS API.

Kaggle exposes a running notebook session through a proxy URL that embeds a
per-session token:

    https://kkb-production.jupyter-proxy.kaggle.net/k/<session>/<token>/proxy

That URL changes every time the session restarts, so it is passed in rather
than committed. Point `KERNEL_URL` at it, or pass `--url`.

    export KERNEL_URL='https://.../proxy'
    python scripts/remote_kernel.py status
    python scripts/remote_kernel.py push src configs scripts
    python scripts/remote_kernel.py run  -f notebooks/src/_bootstrap.py
    python scripts/remote_kernel.py run  'print(1+1)'
    python scripts/remote_kernel.py pull processed/cohort_summary.json

Note the session dies on idle timeout and takes /kaggle/working with it, so
`push` is cheap and expected to be re-run after every reconnect.
"""

import argparse
import base64
import json
import os
import sys
import uuid

import requests

REMOTE_ROOT = os.environ.get("REMOTE_ROOT", "clinical-eegnn")
TEXT_EXT = {".py", ".yaml", ".yml", ".md", ".txt", ".json", ".ipynb", ".cfg", ".toml"}
SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", ".ipynb_checkpoints"}


def base_url(arg):
    url = arg or os.environ.get("KERNEL_URL")
    if not url:
        sys.exit("no kernel URL: pass --url or set KERNEL_URL")
    return url.rstrip("/")


# --------------------------------------------------------------------- status

def cmd_status(base, args):
    for path in ("/api/status", "/api/kernels", "/api/sessions"):
        r = requests.get(base + path, timeout=30)
        print(f"{path:16s} HTTP {r.status_code}  {r.text[:200]}")
        if r.status_code == 404:
            print("\n  404 on every endpoint means the session has ended. "
                  "Restart the Kaggle notebook and re-export KERNEL_URL.")
            return 1
    return 0


# ----------------------------------------------------------------------- push

def _mkdirs(base, remote_dir, session):
    cur = ""
    for part in [p for p in remote_dir.split("/") if p]:
        cur = f"{cur}/{part}" if cur else part
        if session.get(f"{base}/api/contents/{cur}", timeout=30).status_code == 404:
            session.put(f"{base}/api/contents/{cur}", json={"type": "directory"},
                        timeout=30).raise_for_status()


def _put(base, local, remote, session):
    _mkdirs(base, os.path.dirname(remote), session)
    raw = open(local, "rb").read()
    if os.path.splitext(local)[1] in TEXT_EXT:
        body = {"type": "file", "format": "text", "content": raw.decode("utf-8")}
    else:
        body = {"type": "file", "format": "base64",
                "content": base64.b64encode(raw).decode()}
    session.put(f"{base}/api/contents/{remote}", json=body, timeout=120).raise_for_status()
    return len(raw)


def cmd_push(base, args):
    session, total, count = requests.Session(), 0, 0
    for target in args.paths:
        if os.path.isfile(target):
            total += _put(base, target, f"{REMOTE_ROOT}/{target}", session)
            count += 1
            continue
        for dirpath, dirnames, files in os.walk(target):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for f in sorted(files):
                if os.path.splitext(f)[1] not in TEXT_EXT:
                    continue
                local = os.path.join(dirpath, f)
                _put(base, local, f"{REMOTE_ROOT}/{local}", session)
                total += os.path.getsize(local)
                count += 1
    print(f"pushed {count} files, {total/1e3:.1f} kB -> {REMOTE_ROOT}/")
    return 0


def cmd_pull(base, args):
    r = requests.get(f"{base}/api/contents/{args.remote}", timeout=120)
    r.raise_for_status()
    payload = r.json()
    out = args.out or os.path.basename(args.remote)
    if payload["format"] == "base64":
        open(out, "wb").write(base64.b64decode(payload["content"]))
    else:
        content = payload["content"]
        open(out, "w").write(content if isinstance(content, str)
                             else json.dumps(content, indent=1))
    print(f"{args.remote} -> {out}")
    return 0


# ------------------------------------------------------------------------ run

PREAMBLE = """
import importlib, sys
for _m in [k for k in list(sys.modules) if k == "src" or k.startswith("src.")]:
    del sys.modules[_m]
importlib.invalidate_caches()
"""


def cmd_run(base, args):
    import websocket

    kernels = requests.get(f"{base}/api/kernels", timeout=30).json()
    if not kernels:
        sys.exit("no kernel running on the remote server")
    kid = sorted(kernels, key=lambda k: k["last_activity"])[-1]["id"]

    code = open(args.file).read() if args.file else args.code
    if args.reload:
        code = PREAMBLE + code

    ws = websocket.create_connection(
        f"{base.replace('https://', 'wss://', 1)}/api/kernels/{kid}/channels",
        timeout=args.timeout)
    msg_id = uuid.uuid4().hex
    ws.send(json.dumps({
        "header": {"msg_id": msg_id, "username": "remote_kernel",
                   "session": uuid.uuid4().hex, "msg_type": "execute_request",
                   "version": "5.3"},
        "parent_header": {}, "metadata": {},
        "content": {"code": code, "silent": False, "store_history": True,
                    "user_expressions": {}, "allow_stdin": False,
                    "stop_on_error": True},
        "channel": "shell"}))

    status = 0
    while True:
        try:
            msg = json.loads(ws.recv())
        except websocket.WebSocketTimeoutException:
            print("[remote_kernel] timed out waiting for the kernel; "
                  "execution may still be running remotely", file=sys.stderr)
            status = 2
            break
        if msg.get("parent_header", {}).get("msg_id") != msg_id:
            continue
        t, c = msg["msg_type"], msg.get("content", {})
        if t == "stream":
            sys.stdout.write(c.get("text", ""))
            sys.stdout.flush()
        elif t in ("execute_result", "display_data"):
            data = c.get("data", {})
            if "text/plain" in data:
                print(data["text/plain"])
            for mime in ("image/png", "image/jpeg"):
                if mime in data:
                    print(f"[{mime}: {len(data[mime])} b64 chars]")
        elif t == "error":
            print("\n".join(c.get("traceback", [])), file=sys.stderr)
            status = 1
        elif t == "status" and c.get("execution_state") == "idle":
            break
    ws.close()
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="proxy URL (default: $KERNEL_URL)")
    ap.add_argument("--root", help="remote directory to push into "
                                   "(default: $REMOTE_ROOT or clinical-eegnn). "
                                   "Give each parallel experiment its own.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")

    p = sub.add_parser("push"); p.add_argument("paths", nargs="+")
    p = sub.add_parser("pull"); p.add_argument("remote"); p.add_argument("-o", "--out")

    p = sub.add_parser("run")
    p.add_argument("code", nargs="?")
    p.add_argument("-f", "--file")
    p.add_argument("-t", "--timeout", type=int, default=600)
    p.add_argument("--no-reload", dest="reload", action="store_false",
                   help="skip dropping cached src.* modules first")

    args = ap.parse_args()
    if getattr(args, "root", None):
        global REMOTE_ROOT
        REMOTE_ROOT = args.root.rstrip("/")
    base = base_url(args.url)
    return {"status": cmd_status, "push": cmd_push,
            "pull": cmd_pull, "run": cmd_run}[args.cmd](base, args)


if __name__ == "__main__":
    sys.exit(main())
