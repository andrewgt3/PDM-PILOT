#!/usr/bin/env python3
"""
Connect to ABB Robot Web Services (RWS) and verify the controller is reachable.

Uses ROBOT_IP and ROBOT_PORT from .env (or config.abb_rws). Default base URL:
  http://192.168.0.190:80/rw

Run from repo root:
  python3 scripts/connect_abb_rws.py
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

env_file = REPO_ROOT / ".env"
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file)


def main() -> None:
    from config import get_settings

    settings = get_settings()
    rws = settings.abb_rws
    host = os.environ.get("ROBOT_IP", rws.ip_address)
    port = int(os.environ.get("ROBOT_PORT", str(rws.port)))
    base = f"http://{host}:{port}/rw"
    connection_string = base

    print(f"ABB RWS connection string: {connection_string}")
    print(f"Base URL (for requests):     {base}")
    if rws.user:
        print(f"Username:                    {rws.user}")
    print()

    import socket
    import subprocess

    def step1_via_curl() -> bool:
        """Use curl when Python socket is blocked (e.g. errno 65 on some Macs)."""
        url = f"http://{host}:{port}/rw/system"
        auth = []
        if rws.password is not None:
            pw = rws.password.get_secret_value()
            auth = ["-u", f"{rws.user}:{pw}"]
        result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--connect-timeout", "10", *auth, url],
            capture_output=True,
            text=True,
            timeout=15,
        )
        # exit 0 = success; exit 56 = connection reset by peer (TCP connected, then server closed)
        if result.returncode == 0 or result.returncode == 56:
            return True
        return False

    bind_ip = os.environ.get("ROBOT_BIND_IP", "").strip() or None
    if bind_ip:
        print(f"  Binding to local IP: {bind_ip}")
    print("Step 1: TCP reachability (IPv4)...")
    print(f"  Connecting to {host}:{port} ...")
    step1_ok = False
    use_curl_for_http = False
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        if bind_ip:
            sock.bind((bind_ip, 0))
        sock.connect((host, port))
        sock.close()
        print(f"  OK - {host}:{port} is reachable")
        step1_ok = True
    except OSError as e:
        if getattr(e, "errno", None) == 65:
            print(f"  Python socket blocked ({e}). Trying curl fallback...")
            if step1_via_curl():
                print(f"  OK - {host}:{port} reachable via curl (Python TCP blocked on this host)")
                step1_ok = True
                use_curl_for_http = True  # skip urllib in Step 2
        if not step1_ok:
            print(f"  FAILED - {e}")
            print()
            print("Troubleshooting (run from this Mac):")
            print(f"  ping -c 2 {host}")
            print(f"  curl -v --connect-timeout 5 http://{host}:{port}/rw/system")
            sys.exit(1)

    print("Step 2: HTTP GET /rw/system...")
    url = f"{base}/system"
    step2_ok = False
    if use_curl_for_http:
        auth_arg = []
        if rws.password is not None:
            pw = rws.password.get_secret_value()
            auth_arg = ["-u", f"{rws.user}:{pw}"]
        else:
            auth_arg = ["-u", "Default User:"]  # empty password, often required by RWS
        r = subprocess.run(
            ["curl", "-s", "-w", "\nHTTP_CODE:%{http_code}", "--connect-timeout", "10", *auth_arg, url],
            capture_output=True,
            text=True,
            timeout=15,
        )
        out = (r.stdout or "").strip()
        code_part = ""
        if "HTTP_CODE:" in out:
            out, code_part = out.rsplit("HTTP_CODE:", 1)
        print(f"  curl exit {r.returncode} {code_part}".strip())
        if out:
            preview = out[:500] + ("..." if len(out) > 500 else "")
            print(f"Response preview: {preview}")
        if r.returncode == 0 or "200" in code_part or "401" in code_part:
            step2_ok = True
        if not step2_ok and r.returncode == 56:
            print("  (Connection reset by peer - RWS may require different auth. Set ROBOT_USER/ROBOT_PASSWORD in .env.)")
            step2_ok = True
    else:
        try:
            import urllib.request
            import base64
            req = urllib.request.Request(url, method="GET")
            if rws.password is not None:
                pw = rws.password.get_secret_value()
                creds = base64.b64encode(f"{rws.user}:{pw}".encode()).decode()
                req.add_header("Authorization", f"Basic {creds}")
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
                body = resp.read().decode("utf-8", errors="replace")
                print("  OK")
                print(f"GET {url} -> HTTP {status}")
                if body.strip():
                    preview = body.strip()[:500] + ("..." if len(body) > 500 else "")
                    print(f"Response preview: {preview}")
                step2_ok = True
        except OSError as e:
            if getattr(e, "errno", None) == 65:
                print(f"  Python blocked ({e}). Using curl...")
                auth_arg = ["-u", "Default User:"] if rws.password is None else ["-u", f"{rws.user}:{rws.password.get_secret_value()}"]
                r = subprocess.run(
                    ["curl", "-s", "-w", "\nHTTP_CODE:%{http_code}", "--connect-timeout", "10", *auth_arg, url],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                out = (r.stdout or "").strip()
                code_part = ""
                if "HTTP_CODE:" in out:
                    out, code_part = out.rsplit("HTTP_CODE:", 1)
                print(f"  curl exit {r.returncode} {code_part}".strip())
                if out:
                    preview = out[:500] + ("..." if len(out) > 500 else "")
                    print(f"Response preview: {preview}")
                if r.returncode in (0, 56) or "200" in code_part or "401" in code_part:
                    step2_ok = True
        except Exception as e:
            if "Connection reset" in str(e) or "401" in str(e):
                print(f"  Server responded then closed: {e}")
                print("  RWS is up. If you need auth, set ROBOT_USER and ROBOT_PASSWORD in .env.")
                step2_ok = True
            if not step2_ok:
                print(f"  FAILED - {e}")
                sys.exit(1)


if __name__ == "__main__":
    main()
