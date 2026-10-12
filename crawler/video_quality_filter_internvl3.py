#!/usr/bin/env python3
"""
InternVL3 visual filter for the PedX crawler (stage 2).

Uses the native Transformers integration of InternVL (the `OpenGVLab/InternVL3-*-hf`
checkpoints with AutoProcessor / AutoModelForImageTextToText), so no
`trust_remote_code` and no hand-copied image-tiling code.

Each sampled frame is shown to the model with a yes/no question. Instead of parsing
the generated word, the score is read from the next-token distribution:

    score = P(yes) / (P(yes) + P(no))

which is a graded value in [0, 1], so --threshold is meaningful (the old string
match only ever produced 0 or 1). The video score is the mean over frames.

Usage as a one-off check:
    python crawler/video_quality_filter_internvl3.py VIDEO_ID --duration 1500 --model OpenGVLab/InternVL3-1B-hf
"""

import argparse
import os
from typing import Any, Dict, List, Tuple

from filter_common import VisualFilter

# Fix tokenizers parallelism warning when the crawler spawns yt-dlp/ffmpeg subprocesses.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_MODEL = 'OpenGVLab/InternVL3-8B-hf'

PROMPT = ("This image is a frame from a YouTube video. Is it real camera footage (not a video "
          "game, animation or slideshow) of a public street or road, filmed from a vehicle or at "
          "street level, in which road users such as pedestrians or vehicles are visible? "
          "Answer yes or no.")

YES_VARIANTS = ('Yes', 'yes', 'YES', ' Yes', ' yes')
NO_VARIANTS = ('No', 'no', 'NO', ' No', ' no')


def resolve_device(device: str, torch) -> str:
    if device != 'auto':
        return device
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def yes_score(p_yes: float, p_no: float) -> float:
    """Normalised P(yes) over the two answers; 0.0 when the model puts ~no mass on either."""
    total = p_yes + p_no
    return p_yes / total if total > 1e-12 else 0.0


class VideoQualityFilterInternVL3(VisualFilter):
    """Stage-2 filter backed by an InternVL3 vision-language model."""

    backend = 'internvl3'

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = 'auto',
                 threshold: float = 0.5, prompt: str = PROMPT, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.model_name = model_name
        self.prompt = prompt
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
            self.torch = torch
            # Resolve the device BEFORE loading so weights and inputs share one dtype:
            # bfloat16 on accelerators, float32 on CPU.
            self.device = resolve_device(device, torch)
            self.model_dtype = torch.bfloat16 if self.device != 'cpu' else torch.float32
            print(f"Loading {model_name} on {self.device} ...")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, dtype=self.model_dtype).to(self.device).eval()
            self.yes_ids = self._first_token_ids(YES_VARIANTS)
            self.no_ids = self._first_token_ids(NO_VARIANTS)
        except Exception as e:
            # Fail closed: never silently skip the visual stage.
            raise RuntimeError(f"Could not load InternVL3 model '{model_name}': {e}") from e

    def _first_token_ids(self, variants) -> List[int]:
        tokenizer = self.processor.tokenizer
        return sorted({tokenizer.encode(v, add_special_tokens=False)[0] for v in variants})

    def score_frames(self, paths: List[str]) -> Tuple[List[float], List[Dict[str, Any]]]:
        torch = self.torch
        scores, details = [], []
        for path in paths:
            messages = [{'role': 'user', 'content': [{'type': 'image', 'path': path},
                                                     {'type': 'text', 'text': self.prompt}]}]
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                return_tensors='pt').to(self.device, dtype=self.model_dtype)
            with torch.inference_mode():
                logits = self.model(**inputs).logits[0, -1].float()
            probs = torch.softmax(logits, dim=-1)
            p_yes = float(probs[self.yes_ids].sum())
            p_no = float(probs[self.no_ids].sum())
            scores.append(yes_score(p_yes, p_no))
            details.append({'p_yes': round(p_yes, 4), 'p_no': round(p_no, 4)})
        return scores, details


def main():
    ap = argparse.ArgumentParser(description='Run the InternVL3 visual filter on one YouTube video')
    ap.add_argument('video_id')
    ap.add_argument('--duration', type=int, required=True, help='Video duration in seconds')
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    ap.add_argument('--frames', type=int, default=5)
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()
    f = VideoQualityFilterInternVL3(model_name=args.model, device=args.device,
                                    threshold=args.threshold, n_frames=args.frames)
    print(f.check({'id': args.video_id, 'duration_seconds': args.duration,
                   'video_url': f'https://www.youtube.com/watch?v={args.video_id}'}))


if __name__ == '__main__':
    main()
