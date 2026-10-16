# Quality filters

Two stages decide which candidates enter `discovery.csv`. Every decision - accept, reject
or "could not check" - is written to `runs/<run id>/decisions.csv` with its reason, so
nothing is filtered silently. The defaults were checked on a curated and a query-defined set
(see [Validation](#validation)); they are **not validated against hand labels**. For a study,
produce labels with `scripts/evaluate_filter.py` (main [README](../README.md#validating-the-filters)).

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

Backends share [filter_common.py](filter_common.py)'s `VisualFilter`: get frames, score
each one, average, compare to `--threshold`.

### Frame sources (`--frame-source`)

| Source | Frames | Needs | Notes |
|---|---|---|---|
| `stream` | `--frames N` (default 5) at the midpoints of N equal slices of the video (10%, 30%, ... 90%), full resolution | YouTube must serve the media: cookies or a PO-token plugin | yt-dlp (Python API) resolves the stream; ffmpeg seeks through a localhost range proxy ([video_access.py](video_access.py)) |
| `thumbnails` | YouTube's 3 auto-generated frames `mq1`-`mq3` (320x180) | nothing: public `i.ytimg.com` images | positions are not guaranteed (commonly the 25/50/75% marks); timestamps are logged as null; small pedestrians can be missed |
| `auto` (default) | `stream`, then `thumbnails` for the rest of the run once YouTube refuses the stream | - | `decisions.csv` records `frame_source` per verdict |

Why `auto`: without cookies or a PO-token plugin, YouTube served only the first ~6 MB of
the one video tested and refused deeper seeks (HTTP 403); see the README's
[Visual filter access](../README.md#visual-filter-access). With `--frame-source stream`
the check returns `error` (never `accept`) and the run aborts after
`--max-consecutive-errors` consecutive failures. Black letterbox/pillarbox bars are
excluded from the brightness measurement (only when present symmetrically on both sides and
near-pure black, so dark night frames are never cropped).

### YOLO (`--filter yolo`)

Ultralytics detector, YOLO26 by default (`--yolo-model`, `--yolo-conf`). A frame is a
**street scene** if it contains a car, bus, truck, motorcycle, bicycle, traffic light or
stop sign. The video score is the fraction of street-scene frames; the video is accepted
when the score is at least `--threshold` (0.5). With only 3 thumbnail frames that means at
least 2 of 3.

`--require-person` additionally requires a person in a street-scene frame. **It is off by
default**: on 195 curated PedX-Insight videos it cut sensitivity from 87% to 36%, because a
pedestrian shows up in a few sampled frames of only 46% of long driving videos (see
[Validation](#validation)). Turn it on if you would rather lose recall than keep videos where
no pedestrian was sampled. (The earlier "two or more people" rule is gone: it accepted indoor
vlogs.)

### InternVL3 (`--filter internvl3`)

Uses Transformers' native InternVL support (`OpenGVLab/InternVL3-*-hf` checkpoints; no
`trust_remote_code`). Each frame is shown to the model with a yes/no question (is this real
camera footage, not a game or animation, of a public street or road, filmed from a vehicle
or at street level, with road users visible?). The frame score is

```
P(yes) / (P(yes) + P(no))
```

read from the next-token distribution, a graded value in [0, 1], so `--threshold` is
meaningful. The video score is the mean over frames. On real weights the yes and no tokens
carry about 99% of the probability mass, which confirms the token IDs. The prompt is in
[video_quality_filter_internvl3.py](video_quality_filter_internvl3.py) and is part of what
`evaluate_filter.py` should validate.

**Only `InternVL3-1B-hf` was run** (on CUDA, bfloat16). The default `InternVL3-8B-hf`
(~16 GB) has not been run or validated here; expect different score distributions and
re-check `--threshold` before using it.

### Time of day

The median over the sampled frames of mean luma labels a video `night` (<= 65), `day`
(>= 95) or `unknown`. Calibrated on 111 YouTube city-driving videos whose titles carry an
unambiguous cue ("night" vs "morning/afternoon/day"), using their thumbnail frames: night-titled
videos had median brightness 49 (quartiles 36-64), day-titled 120 (110-140). Lit cities at
night are not very dark, so the night cut-off is higher than intuition suggests. With these
values 79% of night-titled videos were labelled night (4% mislabelled day) and 91% of
day-titled day (3% mislabelled night); the rest fall in `unknown`. The labels come from titles
(weak), the two thresholds were chosen on the same 111 videos (a split-half check preferred cut-offs in
the same range: 60-75 / 90-95), and the earlier values (55 / 80) labelled only 62% of the
night-titled videos correctly. Per-frame brightness is logged in `decisions.csv`
(`frame_brightness`) so the thresholds can be re-fit for other footage. Without frames the
label comes from title words (`night`, `sunset`, `noche`, `夜`, ...) or stays `unknown`.

## Validation

**What was measured** (16 October 2026, thumbnail frames, `yolo26n.pt`, `InternVL3-1B-hf`):

* **Positives:** 195 random videos from PedX-Insight's curated `mapping.csv` (videos its
  author kept for analysis). Not hand-checked: some may be unusable in ways the filters should
  catch.
* **Negatives:** 398 search results (`videoDuration=medium`) for ten queries that should *not*
  count as on-profile road-user footage, 40 each: easy (`morning routine vlog`, `cooking recipe
  tutorial`, `podcast interview full episode`, `minecraft survival lets play`, `piano tutorial
  for beginners`) and hard (`Tokyo drone footage 4K`, `city traffic timelapse 4K`, `GTA 5
  driving around the city gameplay`, `train cab ride driver view`, `car interior review
  walkaround`). **These are weak labels:** a query result is not guaranteed to be off-topic
  (several "timelapse" results are real driving videos), so specificity is a lower bound,
  especially for that group.
* Intervals are 95% Wilson intervals. Sensitivity = share of curated videos kept;
  specificity = share of negatives rejected.

**Results, metadata filter + YOLO (the default `--filter yolo` configuration):**

| Configuration | Sensitivity (curated kept) | Specificity (negatives rejected) |
|---|---|---|
| Metadata filter only | 96.9% [93.5-98.6] (189/195) | 42.2% [37.5-47.1] (168/398) |
| YOLO street-scene only | 86.7% [81.2-90.7] (169/195) | 79.1% [74.9-82.8] (315/398) |
| **Metadata + YOLO street-scene (default)** | **84.1% [78.3-88.6] (164/195)** | **89.9% [86.6-92.5] (358/398)** |
| Metadata + YOLO + `--require-person` | 34.9% [28.5-41.8] (68/195) | 94.5% [91.8-96.3] (376/398) |

The six curated videos the metadata filter rejected were all longer than 3 hours.

Negatives rejected by the default configuration, by group: cooking, podcast, game, music 100%
(40/40); vlog 98%; GTA 100%; train cab 100%; drone 98%; timelapse 84%; **car interior / showroom
walkarounds only 20%** (8/40) - indoor-and-exterior car reviews look like street scenes to a
vehicle detector, and a vision-language model or a title keyword is the way to catch them.

**Results, InternVL3-1B-hf** (the same 195 curated videos, but only a stratified **100**
of the negatives, 10 per group: the GPU was shared with another job and CPU inference takes
~11 s per video, so the rest were not scored). Mean P(yes) over the 3 frames; ROC-AUC
**0.959** (curated vs these negatives; median score 0.82 vs 0.06):

| Configuration (same 195 curated / 100 negatives) | Sensitivity | Specificity |
|---|---|---|
| Metadata + YOLO (default `--filter yolo`) | 84.1% [78.3-88.6] (164/195) | 91.0% [83.8-95.2] (91/100) |
| InternVL3 only, threshold 0.5 | 95.4% [91.5-97.6] (186/195) | 90.0% [82.6-94.5] (90/100) |
| **Metadata + InternVL3, threshold 0.5 (default)** | **92.3% [87.7-95.3] (180/195)** | **100% [96.3-100] (100/100)** |
| Metadata + InternVL3, threshold 0.4 | 94.4% [90.2-96.8] (184/195) | 99.0% [94.6-99.8] (99/100) |
| Metadata + InternVL3, threshold 0.6 | 87.2% [81.8-91.2] (170/195) | 100% [96.3-100] (100/100) |

InternVL3 alone rejected car showroom walkarounds 10/10 (YOLO: 2/10) and missed the most
timelapse-query results (4/10 rejected; as noted, some of those are real driving videos); the
metadata filter catches the rest of the hard groups. The 0.5 default sits where sensitivity
is still above 92% and no sampled negative passed; the sweep from 0.1 to 0.9 is smooth
(sensitivity falls from 99.5% to 21%, specificity rises from 60% to 98%), so the choice is
not knife-edge. Treat the specificity as roughly 96-100% (the Wilson interval), not exactly
100%. Positives were scored on the GPU in bfloat16 and negatives on the CPU in float32; the
five videos scored both ways differed by at most 0.006. YOLO and InternVL3 (both at their
defaults) agree on 87% of the 295 videos. **On this evidence InternVL3 is the better
backend when a GPU is available**; YOLO is the cheap CPU option. Only the 1B checkpoint was
run; the default 8B was not.

Reproduce with `scripts/validate_filters.py` (`collect`, `score --internvl3-model
OpenGVLab/InternVL3-1B-hf`, `report`).

**Caveats.** Weak labels (above); three low-resolution frames per video, so these numbers
do not transfer to full-stream frames; the curated set is one research group's selection and
mostly English-titled urban driving; thresholds other than brightness were left at their
defaults rather than tuned; no precision estimate for a real crawl (that depends on what the
search returns). The brightness check also showed that PedX-Insight's own curated
`time_of_day` flags are unreliable as ground truth (e.g. "South Mumbai - Morning Ride" and
"Driving Bel Air" are flagged night; only 2 of 10 night-flagged videos had a night title).

## Failing closed

* A model that cannot load raises an error; the crawler exits instead of continuing
  unfiltered.
* A video that cannot be checked (stream refused with `--frame-source stream`, ffmpeg error,
  model error) is recorded as `error` and not accepted.
* After `--max-consecutive-errors` such errors in a row the run stops and saves its results.

## Installing

```bash
pip install -e ".[yolo]"        # or ".[internvl3]"; stream frames also need ffmpeg on PATH
```

Windows: `winget install ffmpeg`; macOS: `brew install ffmpeg`; Debian/Ubuntu:
`apt install ffmpeg`. Thumbnail frames need neither ffmpeg nor yt-dlp. For a GPU, install a
CUDA build of `torch` first.

Troubleshooting:

* Importing Transformers fails with a torch/torchaudio CUDA-version mismatch: reinstall
  `torchaudio` to match `torch`, or uninstall it (the filters do not use audio).
* InternVL3 hangs while loading: `from_pretrained` contacts the Hugging Face hub even when
  the weights are cached. Once the model is downloaded, set `HF_HUB_OFFLINE=1`.
* A shared GPU: inference can stall for minutes while another job saturates it. The crawler
  has no GPU timeout; run on `--device cpu` (about 11 s per video for the 1B model on 3
  frames) or wait.
* Ultralytics downloads `yolo26n.pt` (5 MB) into the current directory on first use;
  `*.pt` is git-ignored.

## One-off check of a single video

```bash
python crawler/video_quality_filter_yolo.py VIDEO_ID --duration 1500
python crawler/video_quality_filter_internvl3.py VIDEO_ID --duration 1500 --model OpenGVLab/InternVL3-1B-hf
```
