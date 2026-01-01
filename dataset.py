"""
Download CSE_course_RAG dataset from HuggingFace with retry support.

Usage:
    python dataset.py
"""

import os
import time
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, login
except ImportError:
    print("Installing huggingface_hub...")
    os.system("pip install huggingface_hub")
    from huggingface_hub import snapshot_download, login


def download_dataset(max_retries: int = 10, wait_time: int = 60):
    """
    Download dataset from HuggingFace with retry on rate limit.
    
    Args:
        max_retries: Maximum number of retry attempts
        wait_time: Seconds to wait between retries (increases exponentially)
    """
    repo_id = "hatakekksheeshh/CSE_course_RAG"
    target_dir = Path("data")
    
    print(f"Downloading dataset from {repo_id}...")
    print(f"Target directory: {target_dir.absolute()}")
    print("Note: If rate limited, will retry automatically.\n")
    
    for attempt in range(max_retries):
        try:
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                max_workers=2,  # Reduce concurrent downloads to avoid rate limit
                resume_download=True,  # Resume partial downloads
            )
            
            print(f"\n✅ Dataset downloaded successfully!")
            print(f"📁 Data folder is ready at: {target_dir.absolute()}")
            
            # List contents
            print("\nFolder contents:")
            for item in sorted(target_dir.iterdir()):
                if item.name.startswith("."):
                    continue
                item_type = "📁 Folder" if item.is_dir() else "📄 File"
                print(f"  {item_type}: {item.name}")
            
            return downloaded_path
            
        except Exception as e:
            error_msg = str(e)
            
            if "429" in error_msg or "Too Many Requests" in error_msg or "rate limit" in error_msg.lower():
                current_wait = wait_time * (attempt + 1)  # Increase wait time each retry
                print(f"\n⚠️  Rate limited (attempt {attempt + 1}/{max_retries})")
                print(f"⏳ Waiting {current_wait} seconds before retrying...")
                print("   (Downloaded files are cached, will resume from where it stopped)\n")
                time.sleep(current_wait)
            else:
                print(f"\n❌ Error: {e}")
                if attempt < max_retries - 1:
                    print(f"⏳ Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise
    
    print(f"\n❌ Failed after {max_retries} attempts. Please try again later.")
    return None


if __name__ == "__main__":
    # Login to HuggingFace (optional but helps with rate limits)
    print("=" * 50)
    print("CSE Course RAG - Dataset Downloader")
    print("=" * 50)
    
    try:
        login()
        print("Logged in to HuggingFace\n")
    except Exception as e:
        print(f"Not logged in (anonymous download): {e}\n")
    
    download_dataset()
