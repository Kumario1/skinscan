# test_client.py - send one image to the running API, print detections, save a
# box-annotated copy so you can actually see them. Stdlib only (no requests dep).
#   python test_client.py path/to/image.jpg
import base64
import io
import json
import sys
import urllib.request

from PIL import Image, ImageDraw

path = sys.argv[1] if len(sys.argv) > 1 else None
if not path:
    sys.exit("usage: python test_client.py <image.jpg>")

raw = open(path, "rb").read()
payload = json.dumps({"image": base64.b64encode(raw).decode()}).encode()
req = urllib.request.Request(
    "http://localhost:8000/predict", data=payload,
    headers={"Content-Type": "application/json"},
)
out = json.load(urllib.request.urlopen(req, timeout=120))

print(f"{out['count']} detections")
for d in out["detections"][:15]:
    print(f"  {d['label']:14s} {d['score']:.3f}  {d['bbox']}")

img = Image.open(io.BytesIO(raw)).convert("RGB")
draw = ImageDraw.Draw(img)
for d in out["detections"]:
    draw.rectangle(d["bbox"], outline=(255, 0, 0), width=2)
img.save("pred_vis.jpg")
print("annotated image -> pred_vis.jpg")
