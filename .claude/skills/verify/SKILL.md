---
name: verify
summary: Verify the SkinScan production CLI against an isolated local fixture server.
---

# SkinScan CLI verification

- Create images/catalogs in `mktemp -d`; never add binary fixtures.
- Start a `ThreadingHTTPServer` on `127.0.0.1:0` that accepts `POST {"image": base64_jpeg}` and returns SA-RPN `count` plus `detections`.
- Drive the real surface with `python -m src.pipeline.e2e --image ... --out ... --api http://127.0.0.1:<port>/predict ...`.
- Inspect the output directory, parse `analysis.json`, and open all JPEG artifacts with Pillow.
- Probe malformed JSON on a later tile with a pre-seeded output marker; expect exit 1, preserved marker, and no sibling staging directory.
- Capture the server PID from `$!` and stop exactly that process after each run.
