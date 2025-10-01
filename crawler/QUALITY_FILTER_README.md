# Video Quality Filter for PedX Crawler

This document describes the two-stage video quality filtering system integrated into the PedX Crawler to ensure only high-quality street crossing videos are added to the CSV output.

## Overview

The quality filter implements a two-stage approach with two different Stage 2 options:
- **Stage 1**: Fast metadata-based filtering using YouTube API data (same for both filters)
- **Stage 2A**: Micro-clip analysis using YOLO11 object detection (fast, rule-based)
- **Stage 2B**: Micro-clip analysis using InternVL3 vision-language model (sophisticated, AI-powered)

## Filter Types

### YOLO Filter (Default)
- **Analysis Type**: Object detection using YOLO11
- **Speed**: Fast (~2-5 seconds per video)
- **Memory**: Low requirements
- **Accuracy**: Good for detecting objects and basic scene understanding
- **Best For**: Quick processing, resource-constrained environments

### InternVL3 Filter
- **Analysis Type**: Vision-language model with natural language prompts
- **Speed**: Slower (~10-30 seconds per video on CPU)
- **Memory**: High requirements (~16GB for 8B model)
- **Accuracy**: Better context understanding and nuanced analysis
- **Best For**: High-quality filtering, complex scene analysis

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

## Stage 2: Micro-clip Analysis

Both filters follow the same process for micro-clip analysis:
1. **Download Micro-clip**: Download first 3 seconds using yt-dlp
2. **Extract Frames**: Extract 3 frames using ffmpeg
3. **Analysis**: Apply either YOLO11 object detection or InternVL3 vision-language analysis
4. **Scoring**: Convert analysis results to numerical scores

### YOLO11 Object Detection

#### Object Detection Classes
- `person` (class 0)
- `car` (class 2)
- `motorcycle` (class 3)
- `bus` (class 5)
- `truck` (class 7)
- `traffic light` (class 9)

#### Scoring Rules

**Positive Evidence (any frame with these gets score 1.0)**
- **Person + Vehicle**: `person >= 1` AND `(car|bus|truck|motorcycle >= 1)`
- **Multiple People**: `person >= 2` (likely in street)
- **Traffic Light + Activity**: `traffic light >= 1` AND `(person >= 1 OR any vehicle >= 1)`

**Negative Evidence (rejects if present)**
- **No Activity**: `person = 0` AND `vehicles = 0` across all 3 frames

**Final Decision**
- **Accept**: `video_score = 1.0` (any frame has positive evidence)
- **Reject**: `video_score = 0.0` (no positive evidence found)

### InternVL3 Vision-Language Analysis

#### Analysis Process
1. **Frame Analysis**: Each frame is analyzed with InternVL3 using the prompt:
   ```
   "For the analysis of road user behavior, is this video adequate? Be strict. Only answer yes or no. Answer in a single word."
   ```
2. **Response Processing**: Converts "yes"/"no" responses to 1.0/0.0 scores
3. **Threshold Decision**: Accepts videos with max frame score ≥ threshold (default: 0.9)

#### Advantages over YOLO
- **Context Understanding**: Better understanding of scene composition and context
- **Natural Language Reasoning**: Uses language understanding to assess video adequacy
- **Nuanced Analysis**: More sophisticated assessment of road user behavior relevance
- **Flexible Prompting**: Can be easily modified by changing the analysis prompt

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

#### Core Dependencies (Both Filters)
- `opencv-python==4.8.1.78` - Image processing
- `numpy==1.24.3` - Numerical operations
- `yt-dlp==2025.9.5` - YouTube video downloading
- `ffmpeg` - Video processing (system dependency)

#### YOLO Filter Dependencies
- `ultralytics==8.0.196` - YOLO11 model

#### InternVL3 Filter Dependencies
- `torch>=2.0.0` - PyTorch for deep learning
- `transformers>=4.35.0` - HuggingFace transformers
- `Pillow>=9.0.0` - Image processing
- `accelerate>=0.20.0` - Model acceleration
- `einops>=0.8.0` - Tensor operations
- `timm>=1.0.0` - Vision models

## Usage

### Basic Usage

#### YOLO Filter (Default)
```bash
python pedx-crawler.py --enable-quality-filter --filter-type yolo
```

#### InternVL3 Filter
```bash
python pedx-crawler.py --enable-quality-filter --filter-type internvl3
```

### Advanced Usage

#### YOLO Filter with Full Options
```bash
python pedx-crawler.py \
  --cities-file data/cities.txt \
  --output data/outputs/discovery_yolo.csv \
  --per-city 10 \
  --enable-quality-filter \
  --filter-type yolo \
  --max-upload-months 36 \
  --yolo-model yolo11n.pt \
  --temp-dir tmp \
  --verbose
```

#### InternVL3 Filter with Full Options
```bash
python pedx-crawler.py \
  --cities-file data/cities.txt \
  --output data/outputs/discovery_internvl3.csv \
  --per-city 10 \
  --enable-quality-filter \
  --filter-type internvl3 \
  --threshold 0.9 \
  --device cuda \
  --internvl3-model "OpenGVLab/InternVL3-8B" \
  --max-upload-months 36 \
  --temp-dir tmp \
  --verbose
```

### Makefile Shortcuts

#### Quick Testing
```bash
make test-yolo        # Test YOLO filter (3 videos per city)
make test-internvl3   # Test InternVL3 filter (3 videos per city)
```

#### Full Runs
```bash
make run-yolo         # Full run with YOLO filter
make run-internvl3    # Full run with InternVL3 filter
```

### Important: Quality Video Counting

**The crawler now continues searching until it finds the requested number of quality videos.**

- `--per-city 5` means **5 quality videos** that pass the filter, not 5 total videos processed
- If 70% of videos are filtered out, the crawler will process ~17 videos to find 5 quality ones
- The crawler uses pagination and multiple search terms to ensure it finds enough quality content
- This ensures your CSV always contains the exact number of high-quality videos you requested

### Command Line Options

#### Common Options
| Option | Description | Default |
|--------|-------------|---------|
| `--enable-quality-filter` | Enable two-stage quality filtering | Disabled |
| `--filter-type {yolo,internvl3}` | Choose filter type | yolo |
| `--max-upload-months N` | Maximum age of videos in months | 36 |
| `--temp-dir DIR` | Directory for temporary files | tmp |
| `--per-city N` | Maximum quality videos per city | 50 |

#### YOLO-Specific Options
| Option | Description | Default |
|--------|-------------|---------|
| `--yolo-model PATH` | Path to YOLO11 model file | yolo11n.pt |

#### InternVL3-Specific Options
| Option | Description | Default |
|--------|-------------|---------|
| `--threshold FLOAT` | Score threshold for acceptance (0.0-1.0) | 0.9 |
| `--device {auto,cpu,cuda,mps}` | Device for model inference | auto |
| `--internvl3-model NAME` | HuggingFace model name | OpenGVLab/InternVL3-8B |

## File Structure

```
crawler/
├── pedx-crawler.py                    # Main crawler with filter integration
├── video_quality_filter_yolo.py      # YOLO quality filter implementation
├── video_quality_filter_internvl3.py  # InternVL3 quality filter implementation
├── requirements.txt                   # Python dependencies
├── QUALITY_FILTER_README.md          # This comprehensive guide
└── tmp/                              # Temporary files directory
```

## Testing

### Quick Tests with Makefile
```bash
# Test YOLO filter (3 videos per city)
make test-yolo

# Test InternVL3 filter (3 videos per city)
make test-internvl3
```

### Manual Testing
```bash
# Set your YouTube API key
export YOUTUBE_API_KEY=your_api_key_here

# Test YOLO filter
python3 pedx-crawler.py \
  --cities-file data/cities.txt \
  --per-city 5 \
  --enable-quality-filter \
  --filter-type yolo \
  --verbose

# Test InternVL3 filter
python3 pedx-crawler.py \
  --cities-file data/cities.txt \
  --per-city 5 \
  --enable-quality-filter \
  --filter-type internvl3 \
  --device cpu \
  --threshold 0.9 \
  --verbose
```

## Performance Considerations

### Stage 1 (Metadata)
- **Speed**: Very fast (~1ms per video)
- **Quota**: No additional API calls
- **Accuracy**: High for obvious rejects

### Stage 2 Analysis

#### YOLO Filter
- **Speed**: ~2-5 seconds per video
- **Memory**: Low requirements (~1GB RAM)
- **Network**: Downloads 3 seconds of video per video
- **Storage**: Temporary files in `tmp/` directory
- **Accuracy**: High for object detection and basic scene understanding

#### InternVL3 Filter
- **Speed**: ~10-30 seconds per video (CPU), ~2-5 seconds (GPU)
- **Memory**: High requirements (~16GB for 8B model, ~52GB for 26B model)
- **Network**: Downloads 3 seconds of video per video
- **Storage**: Temporary files in `tmp/` directory
- **Accuracy**: Higher for context understanding and nuanced analysis

### Quality Video Counting Impact
- **Processing Time**: May increase significantly if many videos are filtered out
- **API Quota**: Uses more quota as it searches through more videos to find quality ones
- **Example**: If 70% of videos are filtered out, finding 10 quality videos may require processing ~33 total videos
- **Pagination**: Automatically uses YouTube API pagination to search through more results
- **Search Terms**: Cycles through multiple search terms if needed

### Optimization Tips

#### General Tips
1. **Use smaller `--per-city`** values for testing
2. **Clean up `tmp/`** directory regularly
3. **Monitor quota usage** - quality filtering may use more API calls
4. **Adjust `--max-upload-months`** to reduce video age range (recent videos are often higher quality)

#### YOLO Filter Tips
1. **Use YOLO for speed** - much faster than InternVL3
2. **Lower memory requirements** - works on most systems
3. **Good for batch processing** - can handle many videos quickly

#### InternVL3 Filter Tips
1. **Use GPU if available** - significantly faster than CPU
2. **Adjust threshold** - lower threshold (0.7-0.8) for more videos, higher (0.9-0.95) for strict filtering
3. **Consider model size** - 8B model is good balance, 26B model for maximum accuracy
4. **Use CPU for compatibility** - works on any system but slower

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

#### InternVL3 Model Issues
```
Warning: Could not load InternVL3 model: [Error details]
```
**Solutions**:
- **Out of Memory**: Use CPU device or smaller model
- **Download Fails**: Check internet connection and HuggingFace access
- **CUDA Issues**: Use `--device cpu` for compatibility

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
**YOLO Filter**: Use smaller model or reduce batch size:
```bash
python pedx-crawler.py --filter-type yolo --yolo-model yolo11n.pt --per-city 5
```

**InternVL3 Filter**: Use CPU device or smaller model:
```bash
python pedx-crawler.py --filter-type internvl3 --device cpu --per-city 5
```

## Configuration

### Customizing Filter Rules

#### Stage 1 Keywords (Both Filters)
Edit the respective filter files to modify:

**YOLO Filter** (`video_quality_filter_yolo.py`):
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

**InternVL3 Filter** (`video_quality_filter_internvl3.py`):
```python
# Same keywords as YOLO filter
self.positive_keywords = [
    "crosswalk", "zebra crossing", "pedestrian crossing", 
    "jaywalking", "intersection"
]

self.negative_keywords = [
    "compilation", "fails", "meme", "try not to laugh"
]
```

#### Stage 2 Scoring Rules

**YOLO Filter** - Edit `_calculate_frame_score()` in `video_quality_filter_yolo.py`:
```python
def _calculate_frame_score(self, class_counts: Dict[str, int]) -> float:
    # Modify these rules to change scoring logic
    person_count = class_counts.get('person', 0)
    vehicle_count = sum(class_counts.get(vehicle, 0) for vehicle in ['car', 'bus', 'truck', 'motorcycle'])
    traffic_light_count = class_counts.get('traffic light', 0)
    
    # Your custom scoring logic here
    # ...
```

**InternVL3 Filter** - Edit `analysis_prompt` in `video_quality_filter_internvl3.py`:
```python
# Customize the analysis prompt
self.analysis_prompt = "For the analysis of road user behavior, is this video adequate? Be strict. Only answer yes or no. Answer in a single word."

# Or create more specific prompts:
self.analysis_prompt = "Does this video show pedestrians crossing roads with vehicles? Answer yes or no."
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

## Filter Comparison

| Feature | YOLO Filter | InternVL3 Filter |
|---------|-------------|------------------|
| **Analysis Type** | Object detection | Vision-language |
| **Speed** | Fast (~2-5s/video) | Slower (~10-30s/video CPU) |
| **Memory** | Low (~1GB) | High (~16GB for 8B model) |
| **Accuracy** | Good for objects | Better for context |
| **Context Understanding** | Limited | Advanced |
| **Customization** | Rule-based | Prompt-based |
| **Best For** | Quick processing | High-quality filtering |

## Examples

### High-Quality Videos Only (InternVL3)
```bash
python pedx-crawler.py \
  --enable-quality-filter \
  --filter-type internvl3 \
  --threshold 0.95 \
  --per-city 20
```

### Balanced Quality/Quantity (YOLO)
```bash
python pedx-crawler.py \
  --enable-quality-filter \
  --filter-type yolo \
  --per-city 50
```

### Maximum Compatibility (InternVL3 on CPU)
```bash
python pedx-crawler.py \
  --enable-quality-filter \
  --filter-type internvl3 \
  --device cpu \
  --threshold 0.7 \
  --per-city 30
```

## Contributing

To extend the quality filter:

1. **Add new Stage 1 rules** in `_stage1_metadata_filter()` (both filters)
2. **Modify Stage 2 scoring** in `_calculate_frame_score()` (YOLO) or `analysis_prompt` (InternVL3)
3. **Add new object classes** in `yolo_classes` dictionary (YOLO only)
4. **Update tests** using the Makefile targets

## License

This quality filter is part of the PedX Crawler project and follows the same license terms.
