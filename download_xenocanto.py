#!/usr/bin/env python3
"""
Xeno-canto API downloader for bird recordings
Downloads bird sounds from xeno-canto.org using their public API
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib import request, parse
from urllib.error import HTTPError, URLError
import concurrent.futures
from typing import List, Dict, Optional


class XenoCantoDownloader:
    """Download bird recordings from Xeno-canto"""
    
    BASE_URL = "https://xeno-canto.org/api/2/recordings"
    
    def __init__(self, output_dir: str = "datasets/xeno-canto", max_workers: int = 4):
        """
        Args:
            output_dir: Directory to save recordings
            max_workers: Number of parallel downloads
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.downloaded_count = 0
        self.failed_count = 0
    
    def search_recordings(
        self,
        query: Optional[str] = None,
        species: Optional[str] = None,
        country: Optional[str] = None,
        quality: Optional[str] = None,
        recording_type: Optional[str] = None,
        max_results: int = 100
    ) -> List[Dict]:
        """
        Search for recordings using Xeno-canto API
        
        Args:
            query: General search query
            species: Scientific or common name
            country: Country name or code
            quality: Quality rating (A, B, C, D, E)
            recording_type: Type of recording (song, call, etc.)
            max_results: Maximum number of recordings to fetch
            
        Returns:
            List of recording metadata dictionaries
        """
        # Build query string
        query_parts = []
        
        if query:
            query_parts.append(query)
        if species:
            query_parts.append(f'"{species}"')
        if country:
            query_parts.append(f"cnt:{country}")
        if quality:
            query_parts.append(f"q:{quality}")
        if recording_type:
            query_parts.append(f"type:{recording_type}")
        
        if not query_parts:
            query_parts.append("bird")  # Default query
        
        query_string = " ".join(query_parts)
        
        print(f"Searching Xeno-canto for: {query_string}")
        print(f"Max results: {max_results}")
        
        all_recordings = []
        page = 1
        
        while len(all_recordings) < max_results:
            # Build URL with pagination
            params = {
                "query": query_string,
                "page": page
            }
            url = f"{self.BASE_URL}?{parse.urlencode(params)}"
            
            try:
                print(f"Fetching page {page}...", end=" ")
                with request.urlopen(url, timeout=30) as response:
                    data = json.loads(response.read().decode())
                
                recordings = data.get("recordings", [])
                num_pages = data.get("numPages", 1)
                num_recordings = data.get("numRecordings", 0)
                
                print(f"Found {len(recordings)} recordings (total: {num_recordings})")
                
                if not recordings:
                    break
                
                all_recordings.extend(recordings)
                
                # Check if we've reached the last page
                if page >= num_pages:
                    break
                
                page += 1
                time.sleep(0.5)  # Be nice to the API
                
            except (HTTPError, URLError) as e:
                print(f"\nError fetching page {page}: {e}")
                break
            except json.JSONDecodeError as e:
                print(f"\nError parsing response: {e}")
                break
        
        # Limit to max_results
        all_recordings = all_recordings[:max_results]
        
        print(f"\nTotal recordings found: {len(all_recordings)}")
        return all_recordings
    
    def download_recording(self, recording: Dict) -> bool:
        """
        Download a single recording
        
        Args:
            recording: Recording metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        recording_id = recording.get("id")
        file_url = recording.get("file")
        
        if not file_url:
            print(f"No file URL for recording {recording_id}")
            return False
        
        # Ensure URL has https
        if file_url.startswith("//"):
            file_url = "https:" + file_url
        
        # Create filename with metadata
        species = recording.get("en", "unknown").replace(" ", "_")
        country = recording.get("cnt", "unknown")
        quality = recording.get("q", "")
        filename = f"{recording_id}_{species}_{country}_{quality}.mp3"
        
        # Sanitize filename
        filename = "".join(c for c in filename if c.isalnum() or c in "._-")
        output_path = self.output_dir / filename
        
        # Skip if already downloaded
        if output_path.exists():
            print(f"✓ Already downloaded: {filename}")
            return True
        
        try:
            print(f"Downloading: {filename}")
            request.urlretrieve(file_url, output_path)
            
            # Save metadata
            metadata_path = output_path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(recording, f, indent=2)
            
            self.downloaded_count += 1
            return True
            
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            self.failed_count += 1
            # Clean up partial download
            if output_path.exists():
                output_path.unlink()
            return False
    
    def download_recordings(self, recordings: List[Dict]) -> None:
        """
        Download multiple recordings in parallel
        
        Args:
            recordings: List of recording metadata dictionaries
        """
        print(f"\nDownloading {len(recordings)} recordings to {self.output_dir}")
        print(f"Using {self.max_workers} parallel workers\n")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.download_recording, rec) for rec in recordings]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in download task: {e}")
        
        print(f"\n{'='*60}")
        print(f"Download complete!")
        print(f"Successfully downloaded: {self.downloaded_count}")
        print(f"Failed: {self.failed_count}")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Download bird recordings from Xeno-canto",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 100 forest bird recordings
  python download_xenocanto.py --query "forest birds" --max 100
  
  # Download specific species with high quality
  python download_xenocanto.py --species "Turdus merula" --quality A --max 50
  
  # Download bird songs from a specific country
  python download_xenocanto.py --country "Brazil" --type song --max 200
  
  # Download with custom output directory
  python download_xenocanto.py --query "owl" --max 50 --output ./my_owls
  
Quality ratings:
  A - Excellent quality
  B - Good quality
  C - Fair quality
  D - Poor quality
  E - Very poor quality

Recording types:
  song, call, alarm call, flight call, duet, etc.
        """
    )
    
    # Search parameters
    parser.add_argument(
        "--query",
        type=str,
        help="General search query (e.g., 'forest birds', 'nightingale')"
    )
    
    parser.add_argument(
        "--species",
        type=str,
        help="Scientific or common species name (e.g., 'Turdus merula')"
    )
    
    parser.add_argument(
        "--country",
        type=str,
        help="Country name or code (e.g., 'Brazil', 'US')"
    )
    
    parser.add_argument(
        "--quality",
        type=str,
        choices=["A", "B", "C", "D", "E"],
        help="Quality rating (A=best, E=worst)"
    )
    
    parser.add_argument(
        "--type",
        type=str,
        help="Recording type (e.g., 'song', 'call', 'alarm call')"
    )
    
    parser.add_argument(
        "--max",
        type=int,
        default=100,
        help="Maximum number of recordings to download (default: 100)"
    )
    
    # Download parameters
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/xeno-canto",
        help="Output directory (default: datasets/xeno-canto)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)"
    )
    
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Only search and display results, don't download"
    )
    
    args = parser.parse_args()
    
    # Validate that at least one search parameter is provided
    if not any([args.query, args.species, args.country, args.quality, args.type]):
        print("Error: Please provide at least one search parameter")
        print("Use --help for more information")
        sys.exit(1)
    
    # Create downloader
    downloader = XenoCantoDownloader(
        output_dir=args.output,
        max_workers=args.workers
    )
    
    # Search for recordings
    recordings = downloader.search_recordings(
        query=args.query,
        species=args.species,
        country=args.country,
        quality=args.quality,
        recording_type=args.type,
        max_results=args.max
    )
    
    if not recordings:
        print("No recordings found matching your criteria")
        sys.exit(0)
    
    # Display sample results
    print("\nSample recordings:")
    for i, rec in enumerate(recordings[:5], 1):
        print(f"{i}. {rec.get('en', 'Unknown')} - {rec.get('cnt', 'Unknown')} - Quality: {rec.get('q', 'N/A')}")
    
    if len(recordings) > 5:
        print(f"... and {len(recordings) - 5} more")
    
    if args.search_only:
        print("\nSearch-only mode. Use without --search-only to download.")
        sys.exit(0)
    
    # Download recordings
    downloader.download_recordings(recordings)


if __name__ == "__main__":
    main()
