#!/usr/bin/env bash
# Transcribe and diarize every episode in data/audio/ on a CUDA box, then pack the results.
# Safe to rerun after an interruption: finished episodes are skipped.
#
#   export HF_TOKEN=hf_...
#   LIMIT=1 bash scripts/cloud/run.sh   # trial: one episode
#   bash scripts/cloud/run.sh           # everything
set -euo pipefail
cd "$(dirname "$0")/../.."

: "${HF_TOKEN:?export HF_TOKEN=hf_... first (the pyannote models are gated)}"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --extra diarize

# faster-whisper (ctranslate2) only finds the cuBLAS and cuDNN that torch installed if they're
# on the library path.
LIBS=$(uv run python -c 'import nvidia.cublas.lib as b, nvidia.cudnn.lib as d; print(b.__path__[0] + ":" + d.__path__[0])')
export LD_LIBRARY_PATH="$LIBS:${LD_LIBRARY_PATH:-}"

echo "GPU check..."
uv run python - <<'PY'
import numpy as np, torch
from faster_whisper import WhisperModel
assert torch.cuda.is_available(), "torch sees no GPU"
model = WhisperModel("tiny", device="cuda", compute_type="float16")
list(model.transcribe(np.zeros(16000, dtype=np.float32))[0])
print("ok:", torch.cuda.get_device_name(0))
PY

uv run python -m coach.transcribe --all --device cuda --diarize ${LIMIT:+--limit "$LIMIT"} 2>&1 \
  | tee -a transcribe.log || echo "Some episodes failed (see transcribe.log); rerun to retry them."

tar -cf results.tar data/transcripts/raw transcribe.log
echo "results.tar is ready ($(du -h results.tar | cut -f1)). Send it home with: runpodctl send results.tar"
