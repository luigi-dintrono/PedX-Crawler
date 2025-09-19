#!/usr/bin/env python3
"""
PedX Crawler - YouTube Street-Crossing Video Discovery Tool

Discovers YouTube videos of street crossings by city and outputs CSV data
compatible with the PedX pipeline.
"""

import os
import sys
import csv
import argparse
import ssl
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

import requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv


class QuotaTracker:
    """Tracks YouTube API quota usage."""
    
    def __init__(self, daily_limit: int = 10000):
        self.daily_limit = daily_limit
        self.used_quota = 0
        self.search_cost = 100  # Cost per search request
        self.video_details_cost = 1  # Cost per video details request
    
    def add_search_request(self, count: int = 1):
        """Add search request cost to quota usage."""
        self.used_quota += self.search_cost * count
    
    def add_video_details_request(self, count: int = 1):
        """Add video details request cost to quota usage."""
        self.used_quota += self.video_details_cost * count
    
    def get_remaining_quota(self) -> int:
        """Get remaining quota."""
        return max(0, self.daily_limit - self.used_quota)
    
    def get_quota_percentage(self) -> float:
        """Get quota usage percentage."""
        return (self.used_quota / self.daily_limit) * 100
    
    def can_make_search_request(self) -> bool:
        """Check if we can make a search request."""
        return self.get_remaining_quota() >= self.search_cost
    
    def get_status(self) -> str:
        """Get quota status string."""
        remaining = self.get_remaining_quota()
        percentage = self.get_quota_percentage()
        return f"Quota: {self.used_quota:,}/{self.daily_limit:,} ({percentage:.1f}%) - {remaining:,} remaining"


class YouTubeDiscovery:
    """Discovers YouTube street-crossing videos by city."""
    
    def __init__(self, api_key: str, quota_tracker: QuotaTracker = None):
        """Initialize with YouTube API key and optional quota tracker."""
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.quota_tracker = quota_tracker or QuotaTracker()
        
        # Fix SSL issues on macOS
        ssl._create_default_https_context = ssl._create_unverified_context
    
    def search_videos(self, city: str, max_results: int = 50, since_date: str = '2020-01-01T00:00:00Z', use_single_search: bool = True) -> List[Dict[str, Any]]:
        """
        Search for street-crossing videos in a specific city.
        
        Args:
            city: City name to search for
            max_results: Maximum number of results to return
            since_date: ISO 8601 date string for filtering videos published after this date
            use_single_search: If True, use only one search term to save quota (100 units vs 500)
            
        Returns:
            List of video metadata dictionaries
        """
        # Optimize search terms based on quota usage
        if use_single_search:
            # Use only the most effective search term to save quota (100 units instead of 500)
            search_terms = [f"{city} street crossing pedestrian"]
        else:
            # Original multiple search terms (uses 500 units per city)
            search_terms = [
                f"{city} street crossing pedestrian",
                f"{city} crosswalk pedestrian",
                f"{city} traffic light pedestrian",
                f"{city} walking street crossing",
                f"{city} pedestrian crossing street"
            ]
        
        all_videos = []
        
        for search_term in search_terms:
            # Check quota before making request
            if not self.quota_tracker.can_make_search_request():
                print(f"QUOTA WARNING: Not enough quota to search for '{search_term}'. {self.quota_tracker.get_status()}")
                break
                
            try:
                # Search for videos
                search_response = self.youtube.search().list(
                    q=search_term,
                    part='id,snippet',
                    type='video',
                    maxResults=min(max_results, 50),
                    order='relevance',
                    publishedAfter=since_date  # Filter by date
                ).execute()
                
                # Track quota usage
                self.quota_tracker.add_search_request()
                
                for item in search_response.get('items', []):
                    video_id = item['id']['videoId']
                    snippet = item['snippet']
                    
                    # Get additional video details
                    video_details = self._get_video_details(video_id)
                    if video_details:
                        self.quota_tracker.add_video_details_request()
                        video_data = {
                            'id': video_id,
                            'name': snippet['title'],
                            'city': city,
                            'video': video_id,
                            'video_url': f"https://www.youtube.com/watch?v={video_id}",
                            'time_of_day': self._extract_time_of_day(snippet['title']),
                            'start_time': '0:00',  # Default start time
                            'end_time': self._extract_duration(video_details),
                            'region_code': self._extract_region_code(city),
                            'channel_name': snippet['channelTitle'],
                            'channel_url': f"https://www.youtube.com/channel/{snippet['channelId']}"
                        }
                        all_videos.append(video_data)
                        
            except HttpError as e:
                if "quotaExceeded" in str(e):
                    print(f"QUOTA EXCEEDED: Cannot search for '{search_term}'. Please wait 24 hours or request quota increase.")
                    raise e  # Re-raise to stop processing
                else:
                    print(f"Error searching for '{search_term}': {e}")
                    continue
            except Exception as e:
                print(f"Unexpected error searching for '{search_term}': {e}")
                continue
        
        # Remove duplicates based on video ID
        seen_ids = set()
        unique_videos = []
        for video in all_videos:
            if video['id'] not in seen_ids:
                seen_ids.add(video['id'])
                unique_videos.append(video)
        
        return unique_videos[:max_results]
    
    def _get_video_details(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific video."""
        try:
            response = self.youtube.videos().list(
                part='contentDetails,statistics',
                id=video_id
            ).execute()
            
            if response['items']:
                return response['items'][0]
        except Exception as e:
            print(f"Error getting details for video {video_id}: {e}")
        
        return None
    
    def _extract_time_of_day(self, title: str) -> str:
        """Extract time of day from video title."""
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['morning', 'dawn', 'sunrise']):
            return 'morning'
        elif any(word in title_lower for word in ['afternoon', 'midday', 'noon']):
            return 'afternoon'
        elif any(word in title_lower for word in ['evening', 'dusk', 'sunset']):
            return 'evening'
        elif any(word in title_lower for word in ['night', 'nighttime', 'dark']):
            return 'night'
        else:
            return 'unknown'
    
    def _extract_duration(self, video_details: Dict[str, Any]) -> str:
        """Extract duration from video details."""
        duration = video_details.get('contentDetails', {}).get('duration', 'PT0S')
        # Convert ISO 8601 duration to readable format
        # This is a simplified conversion - in production you'd want a proper parser
        if duration.startswith('PT'):
            duration = duration[2:]
            if 'H' in duration:
                hours = duration.split('H')[0]
                minutes = duration.split('H')[1].split('M')[0] if 'M' in duration else '0'
                return f"{hours}:{minutes.zfill(2)}:00"
            elif 'M' in duration:
                minutes = duration.split('M')[0]
                seconds = duration.split('M')[1].split('S')[0] if 'S' in duration else '0'
                return f"0:{minutes.zfill(2)}:{seconds.zfill(2)}"
            else:
                seconds = duration.split('S')[0]
                return f"0:00:{seconds.zfill(2)}"
        return '0:00:00'
    
    def _extract_region_code(self, city: str) -> str:
        """Extract region code from city name."""
        # Simple mapping - in production you'd want a more comprehensive mapping
        city_country_map = {
            'New York': 'US',
            'London': 'GB',
            'Tokyo': 'JP',
            'Paris': 'FR',
            'Berlin': 'DE',
            'Mumbai': 'IN',
            'São Paulo': 'BR',
            'Mexico City': 'MX',
            'Cairo': 'EG',
            'Lagos': 'NG',
            'Sydney': 'AU',
            'Toronto': 'CA',
            'Moscow': 'RU',
            'Istanbul': 'TR',
            'Bangkok': 'TH',
            'Jakarta': 'ID',
            'Manila': 'PH'
        }
        return city_country_map.get(city, 'UNKNOWN')


def convert_date_to_iso(since_date: str) -> str:
    """Convert YYYY-MM-DD date to ISO 8601 format for YouTube API."""
    try:
        # Parse the date and convert to ISO format
        date_obj = datetime.strptime(since_date, '%Y-%m-%d')
        return date_obj.strftime('%Y-%m-%dT00:00:00Z')
    except ValueError:
        print(f"Error: Invalid date format '{since_date}'. Please use YYYY-MM-DD format.")
        sys.exit(1)


def load_cities(file_path: str) -> List[str]:
    """Load cities from text file."""
    cities = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                city = line.strip()
                if city and not city.startswith('#'):
                    # Handle "City,CC" format
                    if ',' in city:
                        city = city.split(',')[0].strip()
                    cities.append(city)
    except FileNotFoundError:
        print(f"Error: Cities file not found at {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading cities file: {e}")
        sys.exit(1)
    
    return cities


def get_unique_filename(output_path: str) -> str:
    """Get a unique filename by adding a number if the file already exists."""
    if not os.path.exists(output_path):
        return output_path
    
    # Split path into directory, filename, and extension
    directory = os.path.dirname(output_path)
    filename = os.path.basename(output_path)
    name, ext = os.path.splitext(filename)
    
    # Find the next available number
    counter = 1
    while True:
        new_filename = f"{name}_{counter}{ext}"
        new_path = os.path.join(directory, new_filename)
        if not os.path.exists(new_path):
            return new_path
        counter += 1


def save_to_csv(videos: List[Dict[str, Any]], output_path: str):
    """Save video data to CSV file with automatic conflict resolution."""
    fieldnames = [
        'id', 'name', 'city', 'video', 'video_url', 'time_of_day',
        'start_time', 'end_time', 'region_code', 'channel_name', 'channel_url'
    ]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get unique filename to avoid overwriting existing files
    unique_path = get_unique_filename(output_path)
    
    with open(unique_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(videos)
    
    return unique_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='PedX Crawler - Discover YouTube street-crossing videos by city')
    parser.add_argument('--api-key', help='YouTube API key')
    parser.add_argument('--api-key-file', help='Path to file containing YouTube API key')
    parser.add_argument('--cities-file', default='data/cities.txt', help='Path to cities file')
    parser.add_argument('--output', default='data/outputs/discovery.csv', help='Output CSV file path')
    parser.add_argument('--max-results', type=int, default=50, help='Maximum results per city (deprecated, use --per-city)')
    parser.add_argument('--per-city', type=int, default=50, help='Maximum results per city')
    parser.add_argument('--since', default='2020-01-01', help='Filter videos published after this date (YYYY-MM-DD format)')
    parser.add_argument('--use-multiple-search', action='store_true', help='Use multiple search terms per city (uses 5x more quota)')
    parser.add_argument('--quota-limit', type=int, default=10000, help='Daily quota limit (default: 10000)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = None
    if args.api_key:
        api_key = args.api_key
    elif args.api_key_file:
        try:
            with open(args.api_key_file, 'r') as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            print(f"Error: API key file not found at {args.api_key_file}")
            sys.exit(1)
    else:
        api_key = os.getenv('YOUTUBE_API_KEY')
    
    if not api_key:
        print("Error: YouTube API key not provided. Use --api-key, --api-key-file, or set YOUTUBE_API_KEY environment variable.")
        sys.exit(1)
    
    # Convert date format
    since_date_iso = convert_date_to_iso(args.since)
    
    # Determine per-city limit (prefer --per-city over --max-results)
    per_city = args.per_city if args.per_city != 50 or args.max_results == 50 else args.max_results
    
    # Initialize quota tracker
    quota_tracker = QuotaTracker(daily_limit=args.quota_limit)
    
    # Load cities
    cities = load_cities(args.cities_file)
    if args.verbose:
        print(f"Loaded {len(cities)} cities: {', '.join(cities)}")
        print(f"Searching for videos published after {args.since} ({since_date_iso})")
        print(f"Maximum {per_city} videos per city")
        print(f"Search mode: {'Multiple terms (5x quota)' if args.use_multiple_search else 'Single term (optimized)'}")
        print(f"Quota limit: {args.quota_limit:,} units")
        print(f"Estimated quota usage: {len(cities) * (500 if args.use_multiple_search else 100):,} units")
    
    # Initialize discovery tool
    discovery = YouTubeDiscovery(api_key, quota_tracker)
    
    # Discover videos for each city
    all_videos = []
    for i, city in enumerate(cities, 1):
        if args.verbose:
            print(f"Searching for videos in {city} ({i}/{len(cities)})...")
            print(f"  {quota_tracker.get_status()}")
        
        try:
            videos = discovery.search_videos(
                city, 
                per_city, 
                since_date_iso, 
                use_single_search=not args.use_multiple_search
            )
            all_videos.extend(videos)
            if args.verbose:
                print(f"  Found {len(videos)} videos for {city}")
                print(f"  {quota_tracker.get_status()}")
        except HttpError as e:
            if "quotaExceeded" in str(e):
                print(f"QUOTA EXCEEDED: Stopping search. {quota_tracker.get_status()}")
                print("Please wait 24 hours for quota reset or request quota increase.")
                break
            else:
                print(f"Error searching videos for {city}: {e}")
                continue
        except Exception as e:
            print(f"Error searching videos for {city}: {e}")
            continue
    
    # Save results
    actual_output_path = save_to_csv(all_videos, args.output)
    print(f"Discovery complete! Found {len(all_videos)} videos total.")
    print(f"Results saved to: {actual_output_path}")
    print(f"Final quota usage: {quota_tracker.get_status()}")


if __name__ == '__main__':
    main()
