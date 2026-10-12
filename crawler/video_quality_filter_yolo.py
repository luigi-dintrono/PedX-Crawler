#!/usr/bin/env python3
"""
YOLO visual filter for the PedX crawler (stage 2).

Samples frames across the whole video (see filter_common.FrameSampler) and runs an
Ultralytics YOLO detector (YOLO26 by default) on them:

  * a frame is a *street scene* when it shows a road vehicle (car, bus, truck,
    motorcycle, bicycle), a traffic light or a stop sign;
  * the video score is the fraction of sampled frames that are street scenes;
  * the video is accepted when score >= threshold (default 0.5) AND at least one
    street-scene frame also contains a person (PedX-Insight analyses pedestrians).

The old "two or more people" rule is gone: it accepted indoor vlogs. Class names are
read from the model, so any checkpoint with COCO-style names works.

Usage as a one-off check:  python crawler/video_quality_filter_yolo.py VIDEO_ID --duration 1500
"""

import argparse
from typing import Any, Dict, List, Tuple

from filter_common import VisualFilter

VEHICLE_CLASSES = frozenset({'car', 'bus', 'truck', 'motorcycle', 'bicycle'})
ROAD_FURNITURE_CLASSES = frozenset({'traffic light', 'stop sign'})


def classify_frame(class_counts: Dict[str, int]) -> Tuple[bool, bool]:
    """(street_scene, pedestrian_in_street) from per-class detection counts."""
    vehicles = sum(class_counts.get(c, 0) for c in VEHICLE_CLASSES)
    furniture = sum(class_counts.get(c, 0) for c in ROAD_FURNITURE_CLASSES)
    street = vehicles > 0 or furniture > 0
    return street, street and class_counts.get('person', 0) > 0


class VideoQualityFilter(VisualFilter):
    """Stage-2 filter backed by an Ultralytics YOLO detector."""

    backend = 'yolo'

    def __init__(self, yolo_model_path: str = 'yolo26n.pt', conf: float = 0.25,
                 threshold: float = 0.5, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.model_name = yolo_model_path
        self.conf = conf
        # Fail closed: a filter that silently skipped this stage would label an
        # unfiltered dataset as filtered.
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(yolo_model_path)
        except Exception as e:
            raise RuntimeError(f"Could not load YOLO model '{yolo_model_path}': {e}") from e

    def score_frames(self, paths: List[str]) -> Tuple[List[float], List[Dict[str, Any]]]:
        results = self.yolo_model(paths, conf=self.conf, verbose=False)
        scores, details = [], []
        for result in results:
            counts: Dict[str, int] = {}
            classes = result.boxes.cls.tolist() if result.boxes is not None else []
            for cls in classes:
                name = result.names[int(cls)]
                counts[name] = counts.get(name, 0) + 1
            street, pedestrian = classify_frame(counts)
            scores.append(1.0 if street else 0.0)
            details.append({'counts': counts, 'street': street, 'pedestrian': pedestrian})
        return scores, details

    def decide(self, scores, frame_details):
        accepted, score, reason = super().decide(scores, frame_details)
        if accepted and not any(d.get('pedestrian') for d in frame_details):
            return False, score, f"street scenes ({reason}) but no person in any street frame"
        return accepted, score, f"street-scene fraction: {reason}"


def main():
    ap = argparse.ArgumentParser(description='Run the YOLO visual filter on one YouTube video')
    ap.add_argument('video_id')
    ap.add_argument('--duration', type=int, required=True, help='Video duration in seconds')
    ap.add_argument('--model', default='yolo26n.pt')
    ap.add_argument('--frames', type=int, default=5)
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()
    f = VideoQualityFilter(yolo_model_path=args.model, threshold=args.threshold, n_frames=args.frames)
    print(f.check({'id': args.video_id, 'duration_seconds': args.duration,
                   'video_url': f'https://www.youtube.com/watch?v={args.video_id}'}))


if __name__ == '__main__':
    main()
