# Quality filters

Two stages decide which candidates enter `discovery.csv`. Every decision - accept, reject
or "could not check" - is written to `runs/<run id>/decisions.csv` with its reason, so
nothing is filtered silently. The rules below are **heuristics that have not yet been
validated against labelled data**; use `scripts/evaluate_filter.py` (see the main
[README](../README.md#validating-the-filters)) before trusting them for a study.

## Stage 1 - metadata filter (always on)

Implemented in [filter_common.py](filter_common.py) (`MetadataFilter`); needs no ML
packages and no downloads. Disable with `--no-metadata-filter`. A candidate is rejected
when any of the following hold (the first matching reason is logged):

| Rule | Detail |
|---|---|
| Live broadcast | `liveBroadcastContent` is `live` or `upcoming` |
| Duration unknown / zero | live streams and still-processing uploads report `PT0S` |
| Too short / too long | outside the profile's window (`dashcam` and `walking`: 240 s - 3 h; `crossing`: 60 s - 20 min); override with `--min-duration` / `--max-duration` |
| 360° video | `contentDetails.projection == "360"` |
| Blocked keyword in title or tags | compilation, fails, meme, prank, reaction, gameplay, GTA, Euro Truck Simulator, BeamNG, Forza, Minecraft, Roblox, simulator, timelapse / time-lapse / hyperlapse, sped up, fast motion, `#shorts`, plus profile extras (`dashcam`: drone, cab ride, cockpit, train ride) |
| Altered playback speed | title patterns such as `3x Speed`, `x4 speed`, `fast forward`, `sped up`: sped-up footage corrupts PedX-Insight's walking and crossing speed estimates |
| Too old (optional) | `--max-upload-months N`, measured from `--until` (or run start) so it is reproducible |
| No profile keyword (optional) | `--require-title-keywords`: the title or tags must contain one of the profile's multilingual keywords |

Keywords match whole words (so "Niagara Falls" is not "fails"); for Chinese, Japanese,
Korean and Thai, which have no word separators, they match as substrings. The old
requirement that every title contain one of five English crossing phrases is gone: it
rejected every non-English title.

Shorts are caught by the `#shorts` tag and, in the `dashcam` and `walking` profiles, by
the 240 s duration floor (Shorts are at most 3 minutes; the `crossing` profile's 60 s floor
does not exclude them). The previous check for "shorts" in the video URL could never fire,
because URLs are always `watch?v=`.

## Stage 2 - visual filter (optional)

Backends share [filter_common.py](filter_common.py)'s `VisualFilter`: sample frames, score
each one, average, compare to `--threshold`.

### Frame sampling

`--frames N` (default 5) frames are taken at the midpoints of N equal slices of the video
(10%, 30%, 50%, 70%, 90% for N=5), not from the first seconds, which are usually intros
and title cards. yt-dlp (Python API) resolves the stream and ffmpeg seeks to each
timestamp through a localhost range-chunking proxy ([video_access.py](video_access.py)).
Temporary frames are deleted after each video.

**YouTube must serve the media**: see the README's
[Visual filter access](../README.md#visual-filter-access). Without cookies or a PO-token
plugin mid-video seeks fail; the check then returns `error` (never `accept`) and the run
aborts after `--max-consecutive-errors` consecutive failures.

### YOLO (`--filter yolo`)

Ultralytics detector, YOLO26 by default (`--yolo-model`, `--yolo-conf`). A frame is a
**street scene** if it contains a car, bus, truck, motorcycle, bicycle, traffic light or
stop sign. The video score is the fraction of street-scene frames; the video is accepted
when the score is at least `--threshold` (0.5) **and** at least one street-scene frame also
contains a person. (The earlier "two or more people" rule accepted indoor vlogs.)

### InternVL3 (`--filter internvl3`)

Uses Transformers' native InternVL support (`OpenGVLab/InternVL3-*-hf` checkpoints; no
`trust_remote_code`). Each frame is shown to the model with a yes/no question (is this real
camera footage, not a game or animation, of a public street or road, filmed from a vehicle
or at street level, with road users visible?). The frame score is

```
P(yes) / (P(yes) + P(no))
```

read from the next-token distribution, a graded value in [0, 1], so `--threshold` is
meaningful. The video score is the mean over frames. Smaller checkpoints
(`InternVL3-1B-hf`, `-2B-hf`) trade accuracy for speed; the prompt is in
[video_quality_filter_internvl3.py](video_quality_filter_internvl3.py) and is part of what
`evaluate_filter.py` should validate.

### Time of day

The median mean-luma of the sampled frames labels a video `night` (<= 55), `day` (>= 80) or
`unknown`. These are uncalibrated heuristics; per-frame brightness is logged in
`decisions.csv` (`frame_brightness`) so they can be recalibrated. Without frames the label
comes from title words (`night`, `sunset`, `noche`, `夜`, ...) or stays `unknown`.

## Failing closed

* A model that cannot load raises an error; the crawler exits instead of continuing
  unfiltered.
* A video that cannot be checked (download refused, ffmpeg error, model error) is recorded
  as `error` and not accepted.
* After `--max-consecutive-errors` such errors in a row the run stops and saves its results.

## Installing

```bash
pip install -e ".[yolo]"        # or ".[internvl3]"; add ffmpeg to PATH
```

Windows: `winget install ffmpeg`; macOS: `brew install ffmpeg`; Debian/Ubuntu:
`apt install ffmpeg`. For a GPU, install a CUDA build of `torch` first. If importing
Transformers fails with a torch/torchaudio CUDA-version mismatch, reinstall `torchaudio`
to match `torch` (or uninstall it; the filter does not use audio).

## One-off check of a single video

```bash
python crawler/video_quality_filter_yolo.py VIDEO_ID --duration 1500
python crawler/video_quality_filter_internvl3.py VIDEO_ID --duration 1500 --model OpenGVLab/InternVL3-1B-hf
```
