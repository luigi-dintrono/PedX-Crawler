#!/usr/bin/env python3
"""
Video Quality Filter for PedX Crawler - InternVL3 Version

Implements a two-stage filtering system:
- Stage 1: Metadata heuristic filtering using YouTube API data (same as YOLO version)
- Stage 2: Micro-clip analysis using InternVL3 vision-language model
"""

import os
import re
import subprocess
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Fix tokenizers parallelism warning when using subprocess
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import cv2
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


class VideoQualityFilterInternVL3:
    """Two-stage video quality filter using InternVL3 for street crossing videos."""
    
    def __init__(self, 
                 max_upload_months: int = 36,
                 model_name: str = "OpenGVLab/InternVL3-8B",
                 temp_dir: str = "tmp",
                 threshold: float = 0.9,
                 device: str = "auto"):
        """
        Initialize the video quality filter with InternVL3.
        
        Args:
            max_upload_months: Maximum age of videos in months
            model_name: HuggingFace model name for InternVL3
            temp_dir: Directory for temporary files
            threshold: Score threshold for acceptance (0.0-1.0)
            device: Device to run model on ('auto', 'cpu', 'cuda', 'mps')
        """
        self.max_upload_months = max_upload_months
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.threshold = threshold
        
        # Initialize InternVL3 model
        try:
            print("Loading InternVL3 model...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16 if device != 'cpu' else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).eval()
            
            # Set device
            if device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            else:
                self.device = device
                
            self.model = self.model.to(self.device)
            print(f"InternVL3 model loaded on {self.device}")
            
        except Exception as e:
            print(f"Warning: Could not load InternVL3 model: {e}")
            self.model = None
            self.tokenizer = None
        
        # Stage 1 criteria (same as YOLO version)
        self.positive_keywords = [
            "crosswalk", "zebra crossing", "pedestrian crossing", 
            "jaywalking", "intersection"
        ]
        
        self.negative_keywords = [
            "compilation", "fails", "meme", "try not to laugh"
        ]
        
        # InternVL3 prompt for road user behavior analysis
        self.analysis_prompt = "For the analysis of road user behavior, is this video adequate? Be strict. Only answer yes or no. Answer in a single word."
    
    def filter_video(self, video_data: Dict) -> Tuple[bool, str]:
        """
        Apply two-stage filtering to a video.
        
        Args:
            video_data: Video metadata dictionary from YouTube API
            
        Returns:
            Tuple of (accepted: bool, reason: str)
        """
        # Stage 1: Metadata heuristic (same as YOLO version)
        stage1_result, stage1_reason = self._stage1_metadata_filter(video_data)
        if not stage1_result:
            return False, f"Stage 1 rejected: {stage1_reason}"
        
        # Stage 2: InternVL3 analysis
        stage2_result, stage2_reason = self._stage2_internvl3_filter(video_data)
        if not stage2_result:
            return False, f"Stage 2 rejected: {stage2_reason}"
        
        return True, "Passed both stages"
    
    def _stage1_metadata_filter(self, video_data: Dict) -> Tuple[bool, str]:
        """Stage 1: Fast metadata-based filtering (same as YOLO version)."""
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
    
    def _stage2_internvl3_filter(self, video_data: Dict) -> Tuple[bool, str]:
        """Stage 2: Micro-clip analysis with InternVL3 (analyzing only 1 frame for speed)."""
        if not self.model or not self.tokenizer:
            return True, "InternVL3 model not available, skipping Stage 2"
        
        video_id = video_data.get('id', '')
        video_url = video_data.get('video_url', '')
        
        if not video_id or not video_url:
            return False, "Missing video ID or URL"
        
        try:
            # Download micro-clip (0-3 seconds)
            clip_path = self._download_micro_clip(video_id, video_url)
            if not clip_path:
                return False, "Failed to download micro-clip"
            
            # Extract only 1 frame (from middle of clip for best representation)
            frame_paths = self._extract_frames(clip_path, video_id)
            if not frame_paths:
                return False, "Failed to extract frames"
            
            # Analyze the single frame with InternVL3
            print(f"    Analyzing 1 frame with InternVL3...")
            video_score = self._analyze_frame_with_internvl3(frame_paths[0])
            
            # Clean up temporary files
            self._cleanup_temp_files(clip_path, frame_paths)
            
            # Decision: accept if frame meets threshold
            if video_score >= self.threshold:
                return True, f"Stage 2 passed with score: {video_score:.3f} (threshold: {self.threshold})"
            else:
                return False, f"Stage 2 failed with score: {video_score:.3f} (threshold: {self.threshold})"
                
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
        """Extract 1 frame from middle of video (1.5 seconds) using ffmpeg."""
        frame_paths = []
        
        try:
            # Extract only 1 frame from the middle of the 3-second clip (at 1.5s)
            frame_path = self.temp_dir / f"{video_id}_01.jpg"
            
            cmd = [
                'ffmpeg',
                '-y',
                '-i', video_path,
                '-ss', '1.5',  # Extract frame at 1.5 seconds (middle of 3-second clip)
                '-vframes', '1',
                '-q:v', '2',  # High quality
                str(frame_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and frame_path.exists():
                frame_paths.append(str(frame_path))
            else:
                print(f"ffmpeg failed for frame extraction: {result.stderr}")
                
        except Exception as e:
            print(f"Error extracting frames: {e}")
        
        return frame_paths
    
    def _analyze_frame_with_internvl3(self, frame_path: str) -> float:
        """Analyze a single frame with InternVL3 and return score."""
        try:
            import time
            
            print(f"      [InternVL3] Loading image: {os.path.basename(frame_path)}")
            # Load image with fewer tiles for faster processing on CPU
            pixel_values = self._load_image(frame_path, max_num=1).to(self.device, dtype=torch.bfloat16 if self.device != 'cpu' else torch.float32)
            
            print(f"      [InternVL3] Running inference (expect 30-90s on CPU)...")
            # Use the model's chat method with proper parameters
            generation_config = {
                'max_new_tokens': 5,  # Reduced from 10 for faster generation
                'do_sample': False,
            }
            
            start_time = time.time()
            
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                self.analysis_prompt,
                generation_config
            )
            
            elapsed = time.time() - start_time
            print(f"      [InternVL3] ✓ Completed in {elapsed:.1f}s - Response: '{response}'")
            
            response = response.strip().lower()
            
            # Convert response to score
            score = self._response_to_score(response)
            
            return score
            
        except Exception as e:
            print(f"Error analyzing frame {frame_path}: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def _load_image(self, image_file, input_size=448, max_num=6):
        """Load and preprocess image for InternVL3."""
        image = Image.open(image_file).convert('RGB')
        transform = self._build_transform(input_size=input_size)
        images = self._dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    def _build_transform(self, input_size):
        """Build image transformation pipeline."""
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform
    
    def _dynamic_preprocess(self, image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
        """Dynamically preprocess image."""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images
    
    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """Find closest aspect ratio."""
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
    
    def _response_to_score(self, response: str) -> float:
        """Convert InternVL3 response to numerical score."""
        response = response.strip().lower()
        
        # Look for "yes" or "no" in the response
        if "yes" in response and "no" not in response:
            return 1.0
        elif "no" in response and "yes" not in response:
            return 0.0
        else:
            # If unclear response, return 0.0 (reject)
            print(f"Unclear response from InternVL3: '{response}'")
            return 0.0
    
    def _extract_duration_seconds(self, video_data: Dict) -> Optional[int]:
        """Extract video duration in seconds from video data (same as YOLO version)."""
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
        """Extract upload date from video data (same as YOLO version)."""
        # This would need to be implemented based on how upload date is stored
        # in the video_data dictionary from the crawler
        # For now, return None (not available in current crawler)
        return None
    
    def _cleanup_temp_files(self, clip_path: str, frame_paths: List[str]):
        """Clean up temporary files (same as YOLO version)."""
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
    filter_instance = VideoQualityFilterInternVL3(threshold=0.9)
    
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
