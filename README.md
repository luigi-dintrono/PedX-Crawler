# PedX Crawler - Road-Crossing Video Discovery

A minimal, production-lean MVP that discovers YouTube street-crossing videos by city and outputs CSV data compatible with the PedX pipeline.

## Features

- 🔍 Discovers YouTube street-crossing videos by city using tuned search terms
- 📊 Outputs CSV with columns: `id`, `name`, `city`, `video`, `video_url`, `time_of_day`, `start_time`, `end_time`, `region_code`, `channel_name`, `channel_url`
- 🔐 Secure API key handling (supports `.env` and `--api-key-file`)
- 🍎 macOS SSL issue fixes included
- 🚀 Easy local and CI execution via Makefile
- ⚡ Quota-optimized search (5x more cities with same API quota)
- 📈 Real-time quota monitoring and management

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

### Makefile targets
- `make install` - Install Python dependencies
- `make setup` - Copy .env.example to .env
- `make run` - Run PedX crawler with default settings
- `make run-verbose` - Run with verbose output
- `make test` - Test with small sample (5 results per city)
- `make test-custom` - Test with custom date and per-city limit
- `make clean` - Clean output files
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

- `google-api-python-client` - YouTube API client
- `python-dotenv` - Environment variable loading
- `requests` - HTTP requests

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

## Project Structure

```
.
├── crawler/
│   ├── pedx-crawler.py           # Main PedX crawler script
│   └── requirements.txt          # Python dependencies
├── data/
│   ├── cities.txt               # Input cities list
│   └── outputs/                 # Output CSV files (gitignored)
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
├── Makefile                     # Build automation
└── README.md                    # This file
```

## License

MIT License - see LICENSE file for details.