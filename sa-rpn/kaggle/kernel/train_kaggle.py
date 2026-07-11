# AcneSCU SA-RPN training on Kaggle (T4/P100). Needs GPU + Internet enabled.
# Mirrors setup_train_thunder.sh; checkpoints land in /kaggle/working/skinscan_out.
# To resume a partial run: attach the previous run's output as an input dataset —
# the script finds the newest epoch_*.pth automatically.
import subprocess
import sys

SCRIPT = r'''
set -euxo pipefail
HOME_DIR=/root
ENV="$HOME_DIR/sa-rpn-env"
REPO="$HOME_DIR/acnedetection"
OUT="/kaggle/working/skinscan_out"
PREPARED="$OUT/coco"
WORK_DIR="$OUT/work_dir"
CKPT="$HOME_DIR/mask_rcnn_r50_fpn_1x_coco.pth"
mkdir -p "$OUT" "$WORK_DIR"

# Kaggle auto-extracts the uploaded zip and mounts it somewhere under /kaggle/input
# (newer images use /kaggle/input/datasets/<owner>/<slug>) - locate by marker file
CONV=$(find /kaggle/input -name 'prepare_acnescu_sa_rpn.py' 2>/dev/null | head -n1)
test -n "$CONV" || { echo "ERROR: dataset not mounted"; find /kaggle/input -maxdepth 3; exit 3; }
DATA=$(dirname "$CONV")
RAW=$(find "$DATA" -maxdepth 1 -type d -name 'AcneSCU.v1*' | head -n1)
test -d "$RAW/train"

# --- 1. legacy env: python 3.8 / torch 1.9 cu111 / mmcv 1.4 -----------------
if [ ! -x "$HOME_DIR/bin/micromamba" ]; then
  mkdir -p "$HOME_DIR/bin"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$HOME_DIR" bin/micromamba
fi
if [ ! -x "$ENV/bin/python" ]; then
  "$HOME_DIR/bin/micromamba" create -y -p "$ENV" python=3.8 pip
  "$ENV/bin/pip" install 'numpy==1.23.5' 'torch==1.9.0+cu111' 'torchvision==0.10.0+cu111' \
      -f https://download.pytorch.org/whl/torch_stable.html
  "$ENV/bin/pip" install 'mmcv-full==1.4.0' \
      -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
  "$ENV/bin/pip" install pycocotools terminaltables scipy pillow matplotlib \
      opencv-python-headless 'yapf==0.31.0' addict pyyaml
fi
PY="$ENV/bin/python"
"$PY" - <<'PYEOF'
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())
print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
assert torch.cuda.is_available(), 'no CUDA device - enable the GPU accelerator in kernel settings'
PYEOF

# --- 2. author code + exact masked crops ------------------------------------
if [ ! -d "$REPO" ]; then
  git clone --depth 1 https://github.com/pingguokiller/acnedetection.git "$REPO"
fi
if [ ! -f "$PREPARED/dataset_metadata.json" ]; then
  "$PY" "$CONV" --voc-root "$RAW" --out "$PREPARED" --test-count 28 --seed 42
fi
cat "$PREPARED/dataset_metadata.json"

# --- 3. patch machine paths / eval target (idempotent), fetch COCO init ------
if [ ! -f "$CKPT" ]; then
  wget -q -O "$CKPT" https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth
fi
test "$(stat -c%s "$CKPT")" -gt 1000000 || { echo "COCO checkpoint truncated"; exit 1; }

CFG_DIR="$REPO/configs/skin" PREPARED="$PREPARED" CKPT="$CKPT" "$PY" - <<'PYEOF'
import os
from pathlib import Path
cfg_dir = Path(os.environ['CFG_DIR']); prepared = os.environ['PREPARED']; ckpt = os.environ['CKPT']
def patch(name, repls):
    p = cfg_dir / name; t = p.read_text()
    for old, new in repls:
        if new in t: continue
        assert old in t, f'{name}: expected text not found:\n  {old}'
        t = t.replace(old, new)
    p.write_text(t)
patch('coco_instance.py', [
    ("data_root = '/home/zhangjw/research/skin_data/coco/'", f"data_root = '{prepared}/'"),
    ("ann_file=data_root + 'annotations/all_whole.json'",    "ann_file=data_root + 'annotations/val_crop_1024.json'"),
    ("img_prefix=data_root + 'images/all_whole/'",           "img_prefix=data_root + 'images/val_crop_1024/'"),
])
patch('default_runtime.py', [('load_from = None', f"load_from = '{ckpt}'")])
sched = (cfg_dir / 'schedule_2x.py').read_text()
for e in ['lr=0.002','momentum=0.9','weight_decay=0.0001','max_epochs=15']:
    assert e in sched, f'schedule_2x.py missing {e}'
print('config patched OK ->', prepared)
PYEOF

# --- 4. train, resuming from an attached previous run if present -------------
RESUME=$(find /kaggle/input -name 'epoch_*.pth' -path '*work_dir*' 2>/dev/null | sort -V | tail -n1 || true)
CMD=("$PY" tools/train.py configs/skin/skin_config.py --work_dir "$WORK_DIR" --seed 42 --deterministic)
if [ -n "${RESUME:-}" ]; then
  echo "Resuming from $RESUME"
  CMD+=(--resume-from "$RESUME")
fi
cd "$REPO"
PYTHONPATH="$REPO" "${CMD[@]}"

echo "DONE. Checkpoints in $WORK_DIR - download from the notebook Output tab."
'''

sys.exit(subprocess.call(['bash', '-c', SCRIPT]))
