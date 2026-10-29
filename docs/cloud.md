# GPU run in the cloud

Transcribe and diarize every episode on a cloud GPU, then bring the transcripts home.
Everything after transcription (speaker ID, correct, distill, ...) runs on the laptop.

There are two ways: **Kaggle** (free, the one in use) and a **rented GPU** (RunPod, paid).

# Kaggle (free)

Kaggle gives about 30 free GPU hours a week, in sessions of up to 12 hours on two T4s.
`coach.cloud.kaggle` uploads the audio once as a private dataset, then pushes a script kernel
per batch, waits for it, downloads the finished transcripts, and pushes the next batch with
whatever is left. Measured on the trial: about 14x realtime per T4 including diarization,
so ~500 hours of audio is about 18 GPU hours, two sessions.

Once: verify your phone number on Kaggle (needed for GPUs and internet), create an API token
(Settings → API), and save it as `~/.kaggle/access_token`. Also do step 0 below (Hugging Face).

```
uv sync --extra pack --extra diarize
uv run python -m coach.cloud.kaggle prepare --limit 2   # trial: two episodes
uv run python -m coach.cloud.kaggle upload
uv run python -m coach.cloud.kaggle run --max-batches 1
uv run python -m coach.cloud.kaggle prepare             # then everything
uv run python -m coach.cloud.kaggle upload
uv run python -m coach.cloud.kaggle run                 # leave the laptop on; hours
```

- The gated pyannote models are copied into the private audio dataset, so Kaggle never gets
  the Hugging Face token. The kernel loads them offline (`HF_HUB_OFFLINE`, `PYANNOTE_CACHE`).
- Kaggle runs `large-v3-turbo` rather than `large-v3`: several times faster on a T4, nearly
  as accurate.
- Each batch stops itself at 11 hours so its outputs are saved; `run` then pushes the next.
  It stops if a batch finishes nothing (quota used up, or a setup error: see the logs in
  `data/cloud/kaggle/out/`).
- `uv run python -m coach.cloud.kaggle status` shows the kernel's state and what's left.
- Episodes downloaded later: `prepare` and `upload` again, then `run`.

Then go to step 8 below.

# Rented GPU (RunPod)

The steps assume RunPod. Any Linux box with an NVIDIA GPU works the same way; only the file
transfer differs (use `scp` on Vast.ai, for example).

**Cost:** the run is roughly 10 to 20 GPU hours on an RTX 4090 (an estimate, not yet
measured). At typical 4090 rates of about $0.35 to $0.70/h that's $5 to $15; check the
current price. The trial run in step 4 gives you the real speed before you commit.

## 0. Once: a Hugging Face token

pyannote's diarization models are free but gated.

1. Make a Hugging Face account and create a **read** token (Settings → Access Tokens).
2. While logged in, open both of these and accept their terms:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Add `HF_TOKEN=hf_...` to `.env` on the laptop. You'll paste it into the GPU box too.

## 1. Pack on the laptop

```
uv sync --extra pack
uv run python -m coach.cloud pack
```

This shrinks each episode to 16 kHz mono Opus (about 5x smaller; Whisper and pyannote use
16 kHz mono anyway), then writes `data/cloud/bundle.tar` with the code and the audio, about
4 to 5 GB. The shrinking takes 1 to 2 hours and reuses files from an earlier run. Don't run it
while a download is going if the laptop runs hot.

## 2. Rent the GPU

On RunPod: **Pods → Deploy**, pick an **RTX 4090** (24 GB), and use the **RunPod PyTorch**
template. Set a **volume disk of at least 30 GB** mounted at `/workspace`, so the work survives
a pod restart. Open the pod's web terminal when it's running.

## 3. Upload

On the laptop, install [runpodctl](https://github.com/runpod/runpodctl/releases) and run:

```
runpodctl send data/cloud/bundle.tar
```

It prints a one-time code. On the pod (runpodctl is preinstalled there):

```
cd /workspace
runpodctl receive <code>
mkdir coach && tar -xf bundle.tar -C coach && cd coach
export HF_TOKEN=hf_...
```

## 4. Trial run: one episode

```
LIMIT=1 bash scripts/cloud/run.sh
```

It installs dependencies, checks that faster-whisper can use the GPU, and transcribes one
episode. The log line `done: ... Nx realtime` gives the speed: total hours of audio (about
530) divided by N is the whole run's length, plus a little for diarization.

## 5. The full run

Run it inside `tmux` so it keeps going if the browser tab closes:

```
tmux new -s run
bash scripts/cloud/run.sh
```

Detach with `Ctrl-b` then `d`; reattach later with `tmux attach -t run`. If anything stops the
run (pod restart, crash), start it again the same way: finished episodes are skipped.

## 6. Bring the results home

When it finishes, `results.tar` holds the transcripts (tens of MB). On the pod:

```
runpodctl send results.tar
```

and on the laptop:

```
runpodctl receive <code>
uv run python -m coach.cloud unpack results.tar
```

## 7. Shut the pod down

**Terminate** the pod (and delete its volume) once the results are unpacked. A stopped pod
still bills for its disk.

## 8. Then, on the laptop

```
uv run python -m coach.speakers                 # who's who; averages Charlie's voiceprint
uv run python -m coach.correct --all
uv run python -m coach.distill --all
uv run python -m coach.consolidate --all
uv run python -m coach.index --rebuild
uv run python -m coach.eval --label "full run"
```

Spot-check speaker ID before trusting it: open a few `data/speakers/<id>.json`, listen to the
episode, and check the hosts are right. Adjust the thresholds in `coach/speakers/identify.py`
if needed, rerun `coach.speakers`, then `coach.distill --all --reattribute` (no LLM calls).

## Troubleshooting

- **`libcudnn` or `libcublas` errors in the GPU check**: ctranslate2's CUDA build doesn't match
  the cuDNN that torch installed. Try `uv pip install "ctranslate2==4.5.0"`, then rerun.
- **Out of GPU memory**: lower whisper's batch with
  `uv run python -m coach.transcribe --all --device cuda --diarize --batch-size 8`.
- **`HF_TOKEN` or 401/403 errors**: the token is wrong, or one of the two model pages in step 0
  wasn't accepted.
