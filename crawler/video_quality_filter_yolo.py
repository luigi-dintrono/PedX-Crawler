#!/usr/bin/env python3
"""
Video Quality Filter for PedX Crawler

Implements a two-stage filtering system:
- Stage 1: Metadata heuristic filtering using YouTube API data
- Stage 2: Micro-clip analysis using YOLO11 object detection
"""

import os
import re
import subprocess
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class VideoQualityFilter:
    """Two-stage video quality filter for street crossing videos."""
    
    def __init__(self, 
                 max_upload_months: int = 36,
                 yolo_model_path: str = "yolo11n.pt",
                 temp_dir: str = "tmp"):
        """
        Initialize the video quality filter.
        
        Args:
            max_upload_months: Maximum age of videos in months
            yolo_model_path: Path to YOLO11 model file
            temp_dir: Directory for temporary files
        """
        self.max_upload_months = max_upload_months
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize YOLO model
        try:
            self.yolo_model = YOLO(yolo_model_path)
        except Exception as e:
            print(f"Warning: Could not load YOLO model: {e}")
            self.yolo_model = None
        
        # Stage 1 criteria
        self.positive_keywords = [
            "crosswalk", "zebra crossing", "pedestrian crossing", 
            "jaywalking", "intersection"
        ]
        
        self.negative_keywords = [
            "compilation", "fails", "meme", "try not to laugh"
        ]
        
        # YOLO class mappings for COCO dataset
        self.yolo_classes = {
            0: 'person',
            2: 'car', 
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
            9: 'traffic light'
        }
        
        # Vehicle classes for co-occurrence detection
        self.vehicle_classes = {2, 3, 5, 7}  # car, motorcycle, bus, truck
    
    def filter_video(self, video_data: Dict) -> Tuple[bool, str]:
        """
        Apply two-stage filtering to a video.
        
        Args:
            video_data: Video metadata dictionary from YouTube API
            
        Returns:
            Tuple of (accepted: bool, reason: str)
        """
        # Stage 1: Metadata heuristic
        stage1_result, stage1_reason = self._stage1_metadata_filter(video_data)
        if not stage1_result:
            return False, f"Stage 1 rejected: {stage1_reason}"
        
        # Stage 2: Micro-clip analysis
        stage2_result, stage2_reason = self._stage2_micro_clip_filter(video_data)
        if not stage2_result:
            return False, f"Stage 2 rejected: {stage2_reason}"
        
        return True, "Passed both stages"
    
    def _stage1_metadata_filter(self, video_data: Dict) -> Tuple[bool, str]:
        """Stage 1: Fast metadata-based filtering."""
        title = video_data.get('name', '').lower()
        
        # Quick rejects (ANY of these conditions)
        # Check for negative keywords
        for keyword in self.negative_keywords:
            if keyword in title:
                return False, f"Title contains negative keyword: '{keyword}'"
        
        # Check for YouTube Shorts (basic check - could be enhanced)
        if 'shorts' in video_data.get('video_url', '').lower():
            return False, "YouTube Short detected"
        
        # Check duration (if available in video_data)
        duration = self._extract_duration_seconds(video_data)
        if duration is not None:
            if duration < 30:  # Less than 30 seconds
                return False, f"Duration too short: {duration}s"
            if duration > 1200:  # More than 20 minutes
                return False, f"Duration too long: {duration}s"
        
        # Check upload date (if available)
        upload_date = self._extract_upload_date(video_data)
        if upload_date:
            months_ago = (datetime.now() - upload_date).days / 30
            if months_ago > self.max_upload_months:
                return False, f"Video too old: {months_ago:.1f} months"
        
        # Quick accepts (ALL must hold)
        # Check for positive keywords
        has_positive_keyword = any(keyword in title for keyword in self.positive_keywords)
        if not has_positive_keyword:
            return False, "No positive keywords found in title"
        
        # Check duration range (if available)
        if duration is not None:
            if not (10 <= duration <= 1200):  # 10s to 20min
                return False, f"Duration outside acceptable range: {duration}s"
        
        return True, "Passed Stage 1 metadata checks"
    
    def _stage2_micro_clip_filter(self, video_data: Dict) -> Tuple[bool, str]:
        """Stage 2: Micro-clip analysis with YOLO11."""
        if not self.yolo_model:
            return True, "YOLO model not available, skipping Stage 2"
        
        video_id = video_data.get('id', '')
        video_url = video_data.get('video_url', '')
        
        if not video_id or not video_url:
            return False, "Missing video ID or URL"
        
        try:
            # Download micro-clip (0-3 seconds)
            clip_path = self._download_micro_clip(video_id, video_url)
            if not clip_path:
                return False, "Failed to download micro-clip"
            
            # Extract 3 frames
            frame_paths = self._extract_frames(clip_path, video_id)
            if not frame_paths:
                return False, "Failed to extract frames"
            
            # Analyze frames with YOLO
            frame_scores = []
            for frame_path in frame_paths:
                score = self._analyze_frame_with_yolo(frame_path)
                frame_scores.append(score)
            
            # Clean up temporary files
            self._cleanup_temp_files(clip_path, frame_paths)
            
            # Decision: accept if any frame has positive evidence
            video_score = max(frame_scores) if frame_scores else 0.0
            
            if video_score >= 1.0:
                return True, f"Stage 2 passed with score: {video_score}"
            else:
                return False, f"Stage 2 failed with score: {video_score}"
                
        except Exception as e:
            return False, f"Stage 2 error: {str(e)}"
    
    def _download_micro_clip(self, video_id: str, video_url: str) -> Optional[str]:
        """Download first 3 seconds of video using yt-dlp."""
        output_path = self.temp_dir / f"{video_id}.mp4"
        
        try:
            # Try system yt-dlp first with robust format selector
            # Use format selector that works better with new YouTube restrictions
            cmd = [
                'yt-dlp',
                '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best[ext=mp4]/best',
                '--download-sections', '*0-3',
                '--max-filesize', '10M',  # Increased from 4M for better success rate
                '--no-check-certificate',
                '--extractor-args', 'youtube:player_client=android',  # Use android client to bypass SABR
                '--user-agent', 'Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36',
                '--merge-output-format', 'mp4',
                '-o', str(output_path),
                video_url
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0 and output_path.exists():
                    return str(output_path)
                else:
                    # Fallback to Python module
                    print(f"System yt-dlp failed, trying Python module for {video_id}")
                    cmd = [
                        'python3', '-m', 'yt_dlp',
                        '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best[ext=mp4]/best',
                        '--download-sections', '*0-3',
                        '--max-filesize', '10M',
                        '--no-check-certificate',
                        '--extractor-args', 'youtube:player_client=android',
                        '--user-agent', 'Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36',
                        '--merge-output-format', 'mp4',
                        '-o', str(output_path),
                        video_url
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    
                    if result.returncode == 0 and output_path.exists():
                        return str(output_path)
                    else:
                        print(f"yt-dlp failed for {video_id}: {result.stderr}")
                        return None
            except FileNotFoundError:
                # System yt-dlp not found, try Python module directly
                print(f"System yt-dlp not found, trying Python module for {video_id}")
                cmd = [
                    'python3', '-m', 'yt_dlp',
                    '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best[ext=mp4]/best',
                    '--download-sections', '*0-3',
                    '--max-filesize', '10M',
                    '--no-check-certificate',
                    '--extractor-args', 'youtube:player_client=android',
                    '--user-agent', 'Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36',
                    '--merge-output-format', 'mp4',
                    '-o', str(output_path),
                    video_url
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0 and output_path.exists():
                    return str(output_path)
                else:
                    print(f"yt-dlp failed for {video_id}: {result.stderr}")
                    return None
                
        except subprocess.TimeoutExpired:
            print(f"yt-dlp timeout for {video_id}")
            return None
        except Exception as e:
            print(f"Error downloading {video_id}: {e}")
            return None
    
    def _extract_frames(self, video_path: str, video_id: str) -> List[str]:
        """Extract 3 frames from video using ffmpeg."""
        frame_paths = []
        
        try:
            for i in range(3):
                frame_path = self.temp_dir / f"{video_id}_{i:02d}.jpg"
                
                cmd = [
                    'ffmpeg',
                    '-y',
                    '-i', video_path,
                    '-ss', str(i),  # Extract frame at second i
                    '-vframes', '1',
                    '-q:v', '2',  # High quality
                    str(frame_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and frame_path.exists():
                    frame_paths.append(str(frame_path))
                else:
                    print(f"ffmpeg failed for frame {i}: {result.stderr}")
                    
        except Exception as e:
            print(f"Error extracting frames: {e}")
        
        return frame_paths
    
    def _analyze_frame_with_yolo(self, frame_path: str) -> float:
        """Analyze a single frame with YOLO11 and return score."""
        try:
            # Run YOLO inference
            results = self.yolo_model(frame_path)
            
            if not results or len(results) == 0:
                return 0.0
            
            # Get detections from first result
            detections = results[0]
            if not hasattr(detections, 'boxes') or detections.boxes is None:
                return 0.0
            
            # Count objects by class
            class_counts = {}
            for box in detections.boxes:
                class_id = int(box.cls[0])
                if class_id in self.yolo_classes:
                    class_name = self.yolo_classes[class_id]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Apply scoring rules
            return self._calculate_frame_score(class_counts)
            
        except Exception as e:
            print(f"Error analyzing frame {frame_path}: {e}")
            return 0.0
    
    def _calculate_frame_score(self, class_counts: Dict[str, int]) -> float:
        """Calculate frame score based on object detection results."""
        person_count = class_counts.get('person', 0)
        vehicle_count = sum(class_counts.get(vehicle, 0) for vehicle in ['car', 'bus', 'truck', 'motorcycle'])
        traffic_light_count = class_counts.get('traffic light', 0)
        
        # Positive evidence rules
        positive_evidence = False
        
        # Rule 1: Person + vehicle co-occurrence
        if person_count >= 1 and vehicle_count >= 1:
            positive_evidence = True
        
        # Rule 2: Multiple people (likely in street)
        if person_count >= 2:
            positive_evidence = True
        
        # Rule 3: Traffic light + (person or vehicle)
        if traffic_light_count >= 1 and (person_count >= 1 or vehicle_count >= 1):
            positive_evidence = True
        
        # Negative evidence
        negative_evidence = False
        
        # Rule 1: No people and no vehicles (likely indoor/other)
        if person_count == 0 and vehicle_count == 0:
            negative_evidence = True
        
        # Return score
        if positive_evidence and not negative_evidence:
            return 1.0
        else:
            return 0.0
    
    def _extract_duration_seconds(self, video_data: Dict) -> Optional[int]:
        """Extract video duration in seconds from video data."""
        # This would need to be implemented based on how duration is stored
        # in the video_data dictionary from the crawler
        duration_str = video_data.get('end_time', '')
        if not duration_str:
            return None
        
        try:
            # Parse duration string (format: "H:MM:SS" or "MM:SS")
            parts = duration_str.split(':')
            if len(parts) == 3:  # H:MM:SS
                hours, minutes, seconds = map(int, parts)
                return hours * 3600 + minutes * 60 + seconds
            elif len(parts) == 2:  # MM:SS
                minutes, seconds = map(int, parts)
                return minutes * 60 + seconds
            else:
                return None
        except (ValueError, IndexError):
            return None
    
    def _extract_upload_date(self, video_data: Dict) -> Optional[datetime]:
        """Extract upload date from video data."""
        # This would need to be implemented based on how upload date is stored
        # in the video_data dictionary from the crawler
        # For now, return None (not available in current crawler)
        return None
    
    def _cleanup_temp_files(self, clip_path: str, frame_paths: List[str]):
        """Clean up temporary files."""
        try:
            if os.path.exists(clip_path):
                os.remove(clip_path)
            for frame_path in frame_paths:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
        except Exception as e:
            print(f"Error cleaning up temp files: {e}")


def test_filter():
    """Test the filter with sample data."""
    filter_instance = VideoQualityFilter()
    
    # Test video data
    test_video = {
        'id': 'test123',
        'name': 'Pedestrian crossing at busy intersection',
        'video_url': 'https://www.youtube.com/watch?v=test123',
        'end_time': '2:30'
    }
    
    result, reason = filter_instance.filter_video(test_video)
    print(f"Filter result: {result}, Reason: {reason}")


if __name__ == '__main__':
    test_filter()

