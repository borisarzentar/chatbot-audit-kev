"""Audit a session JSON through the running local Kev server."""
import argparse
import json
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("state", type=Path, help="JSON file containing the chatbot session state")
parser.add_argument("--url", default="http://127.0.0.1:8009")
args = parser.parse_args()
root = Path(__file__).resolve().parent
payload = {"state": json.loads(args.state.read_text(encoding="utf-8")),
           "questions": json.loads((root / "questions.json").read_text(encoding="utf-8"))}
request = urllib.request.Request(args.url.rstrip("/") + "/v1/systemone",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=300) as response:
    print(json.dumps(json.load(response), indent=2, ensure_ascii=True))
