"""Use the documented Halogen completions route in the private network."""
import hashlib
import http.client
import json
import os
import pathlib
import socket
import sys
import time

TOKENIZER = pathlib.Path("/brewery-tokenizer")
PORT = 28731


def assets(expected):
    actual = sorted(p.name for p in TOKENIZER.iterdir())
    if actual != sorted(pathlib.Path(e["path"]).name for e in expected):
        raise ValueError("private tokenizer directory differs from manifest")
    for entry in expected:
        file = TOKENIZER / pathlib.Path(entry["path"]).name
        if file.is_symlink() or not file.is_file():
            raise ValueError("invalid tokenizer asset")
        with file.open("rb") as stream:
            h = hashlib.sha256()
            for block in iter(lambda: stream.read(1048576), b""):
                h.update(block)
        if h.hexdigest() != entry["sha256"]:
            raise ValueError("private tokenizer SHA-256 mismatch")


def frontend():
    matches = []
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
            if b"/halogen/tools/serve_api.py" not in args:
                continue
            for key, val in ((b"--engine", b"127.0.0.1:28730"),
                             (b"--tokenizer", str(TOKENIZER).encode()),
                             (b"--port", str(PORT).encode())):
                if args.count(key) != 1 or args[args.index(key) + 1] != val:
                    break
            else:
                stat = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                entrypoint = pathlib.Path("/proc") / stat[1]
                if (entrypoint / "stat").read_text().rsplit(")", 1)[1].split()[1] != "1":
                    continue
                if b"/usr/local/bin/entrypoint.sh" not in (entrypoint / "cmdline").read_bytes().split(b"\0"):
                    continue
                socks = set()
                for fd in (entry / "fd").iterdir():
                    try:
                        socks.add(os.readlink(fd))
                    except FileNotFoundError:
                        continue
                listeners = []
                for line in (entry / "net/tcp6").read_text().splitlines()[1:]:
                    cols = line.split()
                    if cols[1] == f"{'0' * 32}:{PORT:04X}" and cols[3] == "0A":
                        listeners.append(cols[9])
                if len(listeners) == 1 and f"socket:[{listeners[0]}]" in socks:
                    matches.append((entry.name, stat[19], listeners[0]))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    if len(matches) != 1:
        raise ValueError("expected one isolated Halogen API frontend owning its listener")
    return matches[0]


def request(prompt, max_tokens):
    connection = http.client.HTTPConnection("::1", PORT, timeout=120)
    try:
        connection.request("POST", "/v1/completions", headers={"Content-Type": "application/json"},
                           body=json.dumps({"prompt": prompt, "model": "brewery-hgn", "n": 1,
                                            "stream": False, "temperature": 0, "max_tokens": max_tokens}))
        response = connection.getresponse()
        data = response.read(262145)
        if response.status != 200 or len(data) > 262144:
            raise ValueError(f"isolated completion failed with HTTP {response.status}")
        result = json.loads(data)
        choices = result["choices"]
        if result["model"] != "brewery-hgn" or len(choices) != 1:
            raise ValueError("isolated completion did not match requested model")
        choice = choices[0]
        text = choice["text"]
        if choice["finish_reason"] not in ("stop", "length") or not isinstance(text, str) or len(text.encode()) > 65536:
            raise ValueError("invalid isolated completion")
        return text
    finally:
        connection.close()


def main():
    job = json.load(sys.stdin)
    assets(job["tokenizer"])
    deadline = time.monotonic() + 120
    while True:
        try:
            identity = frontend()
            with socket.create_connection(("::1", PORT), timeout=1):
                break
        except (OSError, ValueError):
            if time.monotonic() >= deadline:
                raise ValueError("isolated Halogen API did not become ready")
            time.sleep(0.25)
    texts = []
    for i, prompt in enumerate(job["prompts"], 1):
        try:
            texts.append(request(prompt, job["max_tokens"]))
        except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException):
            raise ValueError(f"isolated Halogen completion failed at row {i}") from None
        if frontend() != identity:
            raise ValueError("Halogen API frontend changed during generation")
        print(f"brewery-completed:{i}", file=sys.stderr, flush=True)
    assets(job["tokenizer"])
    print(json.dumps(texts, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from None
