# PedX Crawler - Road-Crossing Video Discovery

A minimal, production-lean MVP that discovers YouTube street-crossing videos by city and outputs CSV data compatible with the PedX pipeline.

## Features

- 🔍 Discovers YouTube street-crossing videos by city using tuned search terms
- 🎯 **Two-stage quality filtering** - Ensures only high-quality videos are added to CSV
- 📊 Outputs CSV with columns: `id`, `name`, `city`, `video`, `video_url`, `time_of_day`, `start_time`, `end_time`, `region_code`, `channel_name`, `channel_url`
- 🔐 Secure API key handling (supports `.env` and `--api-key-file`)
- 🍎 macOS SSL issue fixes included
- 🚀 Easy local and CI execution via Makefile
- ⚡ Quota-optimized search (5x more cities with same API quota)
- 📈 Real-time quota monitoring and management
- 🎯 **Smart video counting** - Continues searching until requested number of quality videos found

## Quick Start

1. **Install dependencies:**
   ```bash
   make install
   ```

2. **Setup environment:**
   ```bash
   make setup
   ```
   Then edit `.env` and add your YouTube API key.

3. **Run PedX crawler:**
   ```bash
   make run
   ```

4. **Run with quality filtering (recommended):**
   ```bash
   make run-filtered
   ```

## API Key Setup

Get your YouTube Data API v3 key from [Google Cloud Console](https://console.developers.google.com/).

### Option 1: Environment file (recommended)
```bash
# Copy the example and edit
cp .env.example .env
# Add your API key to .env
```

### Option 2: Command line
```bash
python3 crawler/pedx-crawler.py --api-key YOUR_API_KEY
```

### Option 3: API key file
```bash
echo "YOUR_API_KEY" > api_key.txt
python3 crawler/pedx-crawler.py --api-key-file api_key.txt
```

## Usage

### Basic usage
```bash
python3 crawler/pedx-crawler.py
```

### With options
```bash
python3 crawler/pedx-crawler.py \
  --cities-file data/cities.txt \
  --output data/outputs/discovery.csv \
  --per-city 50 \
  --since 2024-01-01 \
  --verbose
```

### Quota-optimized mode (recommended)
```bash
# Uses single search term per city (100 units vs 500)
python3 crawler/pedx-crawler.py --since 2024-01-01 --per-city 50 --verbose
```

### High-quality mode (more quota)
```bash
# Uses multiple search terms per city (500 units per city)
python3 crawler/pedx-crawler.py --since 2024-01-01 --per-city 50 --use-multiple-search --verbose
```

### Quality filtering mode (recommended)
```bash
# Enables two-stage quality filtering to ensure only high-quality videos
python3 crawler/pedx-crawler.py --enable-quality-filter --per-city 10 --verbose
```

### Makefile targets
- `make install` - Install Python dependencies
- `make setup` - Copy .env.example to .env
- `make run` - Run PedX crawler with default settings
- `make run-verbose` - Run with verbose output
- `make run-filtered` - Run with quality filter enabled
- `make test` - Test with small sample (5 results per city)
- `make test-filtered` - Test with quality filter (2 videos per city)
- `make test-filtered-recent` - Test with quality filter + recent date
- `make test-custom` - Test with custom date and per-city limit
- `make clean` - Clean output files
- `make clean-temp` - Clean temporary files from quality filter
- `make ci-run` - CI-friendly run (assumes API key in environment)

## Configuration

### Cities file (`data/cities.txt`)
One city per line. Supports both formats:
```
New York
London,GB
Tokyo
```

### Output CSV
The script generates a CSV file with the following columns:
- `id` - YouTube video ID
- `name` - Video title
- `city` - City name
- `video` - Video ID (same as id)
- `video_url` - Full YouTube URL
- `time_of_day` - Extracted from title (morning/afternoon/evening/night/unknown)
- `start_time` - Default start time (0:00)
- `end_time` - Video duration
- `region_code` - Country code (extracted from city)
- `channel_name` - YouTube channel name
- `channel_url` - YouTube channel URL

## Quality Filtering

The PedX Crawler includes an advanced two-stage quality filtering system to ensure only high-quality street crossing videos are added to your dataset.

### Overview

**Stage 1: Metadata Heuristic Filtering** (Fast)
- Duration: 10 seconds - 20 minutes
- Upload date: Within last N months (default: 36)
- Title keywords: Must contain crosswalk, zebra crossing, pedestrian crossing, jaywalking, or intersection
- Rejects: Compilations, fails, memes, shorts, and off-topic content

**Stage 2: Micro-clip YOLO Analysis** (Thorough)
- Downloads first 3 seconds of each video
- Extracts 3 frames using ffmpeg
- Runs YOLO11 object detection
- Detects person + vehicle co-occurrence
- Detects traffic lights + people/vehicles
- Ensures videos show actual street crossing scenarios

### Smart Video Counting

**Important**: The crawler now continues searching until it finds the requested number of **quality videos** that pass the filter, not just total videos processed.

- `--per-city 5` means **5 quality videos** that pass the filter
- If 70% of videos are filtered out, the crawler will process ~17 videos to find 5 quality ones
- Uses pagination and multiple search terms to ensure enough quality content
- Guarantees your CSV contains exactly the number of high-quality videos requested

### Usage

```bash
# Enable quality filtering
python3 crawler/pedx-crawler.py --enable-quality-filter --per-city 10 --verbose

# With recent videos only
python3 crawler/pedx-crawler.py --enable-quality-filter --since 2025-01-01 --per-city 5 --verbose

# Using Makefile
make test-filtered-recent
```

### Dependencies for Quality Filtering

```bash
# Install all dependencies (including quality filter deps)
pip install -r crawler/requirements.txt
brew install ffmpeg  # macOS
```

**Note**: All quality filter dependencies are now included in `requirements.txt`, so a single `pip install -r crawler/requirements.txt` will install everything needed.

### Detailed Documentation

For comprehensive information about the quality filtering system, see:
**[📖 Quality Filter Documentation](crawler/QUALITY_FILTER_README.md)**

## Search Strategy

### Optimized Mode (Default)
The tool uses a single, most effective search term per city:
- `{city} street crossing pedestrian`

**Benefits:**
- 5x more cities with same API quota
- 100 units per city instead of 500
- Real-time quota monitoring

### High-Quality Mode (Optional)
Use `--use-multiple-search` for comprehensive search:
- `{city} street crossing pedestrian`
- `{city} crosswalk pedestrian`
- `{city} traffic light pedestrian`
- `{city} walking street crossing`
- `{city} pedestrian crossing street`

Results are filtered to:
- Recent videos (configurable with `--since`)
- Ordered by relevance
- Duplicates removed
- Limited per city (configurable with `--per-city`)

## Quota Management

### Quota Tracking
- Real-time quota monitoring
- Automatic stopping before quota exceeded
- Detailed usage reporting
- Configurable daily limits

### Quota Usage Comparison
| Mode | Units per City | 10 Cities | 20 Cities |
|------|----------------|-----------|-----------|
| **Optimized** (default) | 100 | 1,000 | 2,000 |
| **Multiple terms** | 500 | 5,000 | 10,000 |

### Quota Increase
Request quota increases in [Google Cloud Console](https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas).

## Requirements

- Python 3.7+
- YouTube Data API v3 key
- Internet connection

## Dependencies

### Core Dependencies
- `google-api-python-client` - YouTube API client
- `python-dotenv` - Environment variable loading
- `requests` - HTTP requests

### Quality Filter Dependencies (Optional)
- `ultralytics` - YOLO11 object detection
- `opencv-python` - Image processing
- `numpy` - Numerical operations
- `yt-dlp` - YouTube video downloading
- `ffmpeg` - Video processing (system dependency)

## Troubleshooting

### macOS SSL Issues
The script automatically fixes common macOS SSL certificate issues.

### API Quota
YouTube API has daily quotas. The script includes:
- Real-time quota monitoring
- Automatic stopping before quota exceeded
- Optimized search mode (5x more cities)
- Detailed quota usage reporting

### Rate Limiting
The script includes error handling for rate limits and API errors.

### Quality Filter Issues
- **YOLO model not found**: The model will be downloaded automatically on first use
- **yt-dlp not found**: Install all dependencies with `pip install -r crawler/requirements.txt`
- **ffmpeg not found**: Install with `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Ubuntu)
- **Memory issues**: Use smaller `--per-city` values or disable Stage 2 filtering
- **Slow processing**: Quality filtering may take longer as it processes more videos to find quality ones

## Project Structure

```
.
├── crawler/
│   ├── pedx-crawler.py              # Main PedX crawler script
│   ├── video_quality_filter.py      # Two-stage quality filtering system
│   ├── requirements.txt             # Python dependencies
│   └── QUALITY_FILTER_README.md     # Detailed quality filter documentation
├── data/
│   ├── cities.txt                   # Input cities list
│   └── outputs/                     # Output CSV files (gitignored)
├── .env.example                     # Environment template
├── .gitignore                       # Git ignore rules
├── Makefile                         # Build automation
└── README.md                        # This file
```

## License

MIT License - see LICENSE file for details.