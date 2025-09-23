# Video Quality Filter for PedX Crawler

This document describes the two-stage video quality filtering system integrated into the PedX Crawler to ensure only high-quality street crossing videos are added to the CSV output.

## Overview

The quality filter implements a two-stage approach:
- **Stage 1**: Fast metadata-based filtering using YouTube API data
- **Stage 2**: Micro-clip analysis using YOLO11 object detection

## Stage 1: Metadata Heuristic Filtering

### Quick Accepts (ALL must hold)
- **Duration**: 10 seconds ≤ duration ≤ 20 minutes
- **Upload Date**: Within last N months (default: 36)
- **Title Keywords**: Must contain any of:
  - "crosswalk"
  - "zebra crossing" 
  - "pedestrian crossing"
  - "jaywalking"
  - "intersection"
- **Video Type**: NOT a YouTube Short (reject vertical videos)

### Quick Rejects (ANY will reject)
- **Title Keywords**: Contains any of:
  - "compilation"
  - "fails"
  - "meme"
  - "try not to laugh"
- **Video Type**: YouTube Shorts or vertical videos
- **Duration**: Less than 30 seconds (stricter than accept lower bound)
- **Channel Category**: Off-topic channels (optional)

## Stage 2: Micro-clip Analysis with YOLO11

### Process
1. **Download Micro-clip**: Download first 3 seconds using yt-dlp
2. **Extract Frames**: Extract 3 frames using ffmpeg
3. **Object Detection**: Run YOLO11 on each frame
4. **Scoring**: Apply positive/negative evidence rules

### Object Detection Classes
- `person` (class 0)
- `car` (class 2)
- `motorcycle` (class 3)
- `bus` (class 5)
- `truck` (class 7)
- `traffic light` (class 9)

### Scoring Rules

#### Positive Evidence (any frame with these gets score 1.0)
- **Person + Vehicle**: `person >= 1` AND `(car|bus|truck|motorcycle >= 1)`
- **Multiple People**: `person >= 2` (likely in street)
- **Traffic Light + Activity**: `traffic light >= 1` AND `(person >= 1 OR any vehicle >= 1)`

#### Negative Evidence (rejects if present)
- **No Activity**: `person = 0` AND `vehicles = 0` across all 3 frames
- **Extreme Blur/Darkness**: Optional OpenCV variance/histogram check

#### Final Decision
- **Accept**: `video_score = 1.0` (any frame has positive evidence)
- **Reject**: `video_score = 0.0` (no positive evidence found)

## Installation

### Prerequisites
```bash
# Install all Python dependencies (including quality filter deps)
pip install -r requirements.txt

# Install ffmpeg for video processing
# macOS:
brew install ffmpeg

# Ubuntu/Debian:
sudo apt install ffmpeg
```

**Note**: All Python dependencies including `yt-dlp` are now included in `requirements.txt`.

### Dependencies
- `ultralytics==8.0.196` - YOLO11 model
- `opencv-python==4.8.1.78` - Image processing
- `numpy==1.24.3` - Numerical operations
- `yt-dlp==2025.9.5` - YouTube video downloading
- `ffmpeg` - Video processing (system dependency)

## Usage

### Basic Usage
```bash
python pedx-crawler.py --enable-quality-filter
```

### Full Options
```bash
python pedx-crawler.py \
  --cities-file data/cities.txt \
  --output data/outputs/discovery_filtered.csv \
  --per-city 10 \
  --enable-quality-filter \
  --max-upload-months 36 \
  --yolo-model yolo11n.pt \
  --temp-dir tmp \
  --verbose
```

### Important: Quality Video Counting

**The crawler now continues searching until it finds the requested number of quality videos.**

- `--per-city 5` means **5 quality videos** that pass the filter, not 5 total videos processed
- If 70% of videos are filtered out, the crawler will process ~17 videos to find 5 quality ones
- The crawler uses pagination and multiple search terms to ensure it finds enough quality content
- This ensures your CSV always contains the exact number of high-quality videos you requested

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--enable-quality-filter` | Enable two-stage quality filtering | Disabled |
| `--max-upload-months N` | Maximum age of videos in months | 36 |
| `--yolo-model PATH` | Path to YOLO11 model file | yolo11n.pt |
| `--temp-dir DIR` | Directory for temporary files | tmp |

## File Structure

```
crawler/
├── pedx-crawler.py              # Main crawler with filter integration
├── video_quality_filter.py      # Quality filter implementation
├── test_filter_integration.py   # Test script
├── example_usage.py             # Usage examples
├── requirements.txt             # Python dependencies
└── QUALITY_FILTER_README.md     # This file
```

## Testing

### Run Basic Tests
```bash
python3 test_filter_integration.py
```

### Test with Real Videos
```bash
# Set your YouTube API key
export YOUTUBE_API_KEY=your_api_key_here

# Run crawler with quality filter on small dataset
python3 pedx-crawler.py \
  --cities-file data/cities.txt \
  --per-city 5 \
  --enable-quality-filter \
  --verbose
```

## Performance Considerations

### Stage 1 (Metadata)
- **Speed**: Very fast (~1ms per video)
- **Quota**: No additional API calls
- **Accuracy**: High for obvious rejects

### Stage 2 (YOLO Analysis)
- **Speed**: Slower (~2-5 seconds per video)
- **Network**: Downloads 3 seconds of video per video
- **Storage**: Temporary files in `tmp/` directory
- **Accuracy**: High for street crossing detection

### Quality Video Counting Impact
- **Processing Time**: May increase significantly if many videos are filtered out
- **API Quota**: Uses more quota as it searches through more videos to find quality ones
- **Example**: If 70% of videos are filtered out, finding 10 quality videos may require processing ~33 total videos
- **Pagination**: Automatically uses YouTube API pagination to search through more results
- **Search Terms**: Cycles through multiple search terms if needed

### Optimization Tips
1. **Use Stage 1 only** for initial filtering if Stage 2 is too slow
2. **Adjust `--max-upload-months`** to reduce video age range (recent videos are often higher quality)
3. **Use smaller `--per-city`** values for testing
4. **Clean up `tmp/`** directory regularly
5. **Monitor quota usage** - quality filtering may use more API calls
6. **Consider `--since` parameter** to focus on recent, higher-quality videos

## Troubleshooting

### Common Issues

#### YOLO Model Not Found
```
Warning: Could not load YOLO model: [Errno 2] No such file or directory: 'yolo11n.pt'
```
**Solution**: The model will be downloaded automatically on first use, or download manually:
```bash
wget https://github.com/ultralytics/assets/releases/download/v8.0.0/yolo11n.pt
```

#### yt-dlp Not Found
```
yt-dlp not found
```
**Solution**: Install all dependencies:
```bash
pip install -r requirements.txt
```

#### ffmpeg Not Found
```
ffmpeg not found
```
**Solution**: Install ffmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

#### Memory Issues
**Solution**: Reduce batch size or use smaller YOLO model:
```bash
python pedx-crawler.py --yolo-model yolo11n.pt --per-city 5
```

## Configuration

### Customizing Filter Rules

Edit `video_quality_filter.py` to modify:

#### Stage 1 Keywords
```python
# Positive keywords (must contain at least one)
self.positive_keywords = [
    "crosswalk", "zebra crossing", "pedestrian crossing", 
    "jaywalking", "intersection"
]

# Negative keywords (reject if contains any)
self.negative_keywords = [
    "compilation", "fails", "meme", "try not to laugh"
]
```

#### Stage 2 Scoring Rules
```python
def _calculate_frame_score(self, class_counts: Dict[str, int]) -> float:
    # Modify these rules to change scoring logic
    person_count = class_counts.get('person', 0)
    vehicle_count = sum(class_counts.get(vehicle, 0) for vehicle in ['car', 'bus', 'truck', 'motorcycle'])
    traffic_light_count = class_counts.get('traffic light', 0)
    
    # Your custom scoring logic here
    # ...
```

## Output

### Filtered Results
Videos that pass both stages are added to the CSV with additional metadata:
- Original video data from YouTube API
- Filter status and reason
- Quality score (if available)

### Logging
With `--verbose` flag, you'll see:
```
✓ Quality filter passed: 'Pedestrian crossing at busy intersection...'
  Filtered out video 'Funny compilation of fails...': Stage 1 rejected: Title contains negative keyword: 'compilation'
```

## Contributing

To extend the quality filter:

1. **Add new Stage 1 rules** in `_stage1_metadata_filter()`
2. **Modify Stage 2 scoring** in `_calculate_frame_score()`
3. **Add new object classes** in `yolo_classes` dictionary
4. **Update tests** in `test_filter_integration.py`

## License

This quality filter is part of the PedX Crawler project and follows the same license terms.
