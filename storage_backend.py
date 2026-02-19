"""
Storage Backend Abstraction Layer
Supports: Supabase + Cloudflare R2, Local filesystem
"""

import os
import json
import requests
import hashlib
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from pathlib import Path
import io

# Supabase and R2 (boto3 for S3-compatible API)
try:
    from supabase import create_client, Client
    import boto3
    from botocore.exceptions import ClientError
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False


# ==================================================
# ABSTRACT BASE CLASS
# ==================================================

class StorageBackend(ABC):
    """Abstract base class for all storage backends"""
    
    @abstractmethod
    def exists(self, reel_id: str) -> bool:
        """Check if reel data exists in cache"""
        pass
    
    @abstractmethod
    def get_metadata(self, reel_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached reel metadata (caption, hashtags, etc.)"""
        pass
    
    @abstractmethod
    def save_reel_data(self, reel_id: str, metadata: Dict[str, Any], video_path: Optional[str] = None) -> bool:
        """
        Save reel data to storage
        Args:
            reel_id: Unique reel identifier
            metadata: Dict with caption, hashtags, location, etc.
            video_path: Local path to video file (optional)
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def get_video_url(self, reel_id: str) -> Optional[str]:
        """Get shareable URL for cached video"""
        pass


# ==================================================
# SUPABASE + CLOUDFLARE R2 IMPLEMENTATION
# ==================================================

class SupabaseR2Storage(StorageBackend):
    """
    Hybrid storage using Supabase (PostgreSQL) for metadata and Cloudflare R2 for videos
    
    Architecture:
    - Supabase table 'reel_cache' stores JSON metadata
    - Cloudflare R2 bucket stores video.mp4 files
    - R2 uses S3-compatible API (boto3)
    """
    
    def __init__(
        self,
        supabase_url: str = None,
        supabase_key: str = None,
        r2_account_id: str = None,
        r2_access_key: str = None,
        r2_secret_key: str = None,
        r2_bucket: str = None,
        r2_endpoint: str = None,
        r2_public_url: str = None
    ):
        """
        Initialize Supabase + R2 storage
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase service_role key (not anon key)
            r2_account_id: Cloudflare account ID
            r2_access_key: R2 API access key
            r2_secret_key: R2 API secret key
            r2_bucket: R2 bucket name
            r2_endpoint: R2 endpoint URL
            r2_public_url: Optional public bucket URL for direct access
        """
        if not SUPABASE_AVAILABLE:
            raise ImportError(
                "Supabase + R2 backend requires: pip install supabase boto3"
            )
        
        # Get config from env if not provided
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_SERVICE_KEY")
        self.r2_account_id = r2_account_id or os.getenv("R2_ACCOUNT_ID")
        self.r2_access_key = r2_access_key or os.getenv("R2_ACCESS_KEY_ID")
        self.r2_secret_key = r2_secret_key or os.getenv("R2_SECRET_ACCESS_KEY")
        self.r2_bucket = r2_bucket or os.getenv("R2_BUCKET_NAME")
        self.r2_endpoint = r2_endpoint or os.getenv("R2_ENDPOINT_URL")
        self.r2_public_url = r2_public_url or os.getenv("R2_PUBLIC_URL")
        
        # Validate required config
        if not all([self.supabase_url, self.supabase_key]):
            raise ValueError(
                "Missing Supabase config. Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env"
            )
        
        if not all([self.r2_access_key, self.r2_secret_key, self.r2_bucket, self.r2_endpoint]):
            raise ValueError(
                "Missing R2 config. Set R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, "
                "R2_BUCKET_NAME, R2_ENDPOINT_URL in .env"
            )
        
        # Initialize Supabase client
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Initialize R2 client (S3-compatible via boto3)
        self.s3 = boto3.client(
            's3',
            endpoint_url=self.r2_endpoint,
            aws_access_key_id=self.r2_access_key,
            aws_secret_access_key=self.r2_secret_key,
            region_name='auto'  # R2 uses 'auto' for region
        )
        
        print(f"✅ Supabase connected: {self.supabase_url}")
        print(f"✅ R2 bucket connected: {self.r2_bucket}")
    
    def exists(self, reel_id: str) -> bool:
        """Check if reel exists in Supabase cache"""
        try:
            response = self.supabase.table("reel_cache").select("reel_id").eq("reel_id", reel_id).execute()
            return len(response.data) > 0
        except Exception as e:
            print(f"❌ Error checking cache: {e}")
            return False
    
    def get_metadata(self, reel_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve reel metadata from Supabase"""
        try:
            response = self.supabase.table("reel_cache").select("*").eq("reel_id", reel_id).execute()
            
            if not response.data:
                return None
            
            # Convert Supabase row to our metadata format
            row = response.data[0]
            
            metadata = {
                "url": row.get("url"),
                "reel_id": row.get("reel_id"),
                "caption": row.get("caption"),
                "hashtags": row.get("hashtags", []),
                "location": row.get("location"),
                "likes": row.get("likes", 0),
                "timestamp": row.get("timestamp"),
                "owner_username": row.get("owner_username"),
                "video_url": row.get("video_url"),
                "display_url": row.get("display_url"),
                "r2_video_key": row.get("r2_video_key"),
                "_from_cache": True,
                "_cached_at": row.get("created_at")
            }
            
            return metadata
            
        except Exception as e:
            print(f"❌ Error retrieving from Supabase: {e}")
            return None
    
    def save_reel_data(self, reel_id: str, metadata: Dict[str, Any], video_path: Optional[str] = None) -> bool:
        """
        Save reel data to Supabase (metadata) + R2 (video)
        
        Args:
            reel_id: Unique reel identifier
            metadata: Dict with caption, hashtags, etc.
            video_path: Local path to video file
        
        Returns:
            True if successful
        """
        try:
            # Step 1: Upload video to R2 (if provided)
            r2_video_key = None
            if video_path and os.path.exists(video_path):
                r2_video_key = f"{reel_id}.mp4"
                
                print(f"💾 Uploading video to R2: {r2_video_key}")
                
                with open(video_path, 'rb') as video_file:
                    self.s3.upload_fileobj(
                        video_file,
                        self.r2_bucket,
                        r2_video_key,
                        ExtraArgs={
                            'ContentType': 'video/mp4',
                            'CacheControl': 'public, max-age=31536000'  # Cache for 1 year
                        }
                    )
                
                print(f"✅ Video uploaded to R2: {r2_video_key}")
            
            # Step 2: Prepare Supabase row
            supabase_row = {
                "reel_id": reel_id,
                "url": metadata.get("url"),
                "caption": metadata.get("caption"),
                "hashtags": metadata.get("hashtags", []),
                "location": metadata.get("location"),
                "likes": metadata.get("likes", 0),
                "timestamp": metadata.get("timestamp"),
                "owner_username": metadata.get("owner_username"),
                "video_url": metadata.get("video_url"),  # Original Apify URL
                "display_url": metadata.get("display_url"),
                "r2_video_key": r2_video_key  # Key in R2 bucket
            }
            
            # Step 3: Upsert to Supabase (insert or update if exists)
            response = self.supabase.table("reel_cache").upsert(
                supabase_row,
                on_conflict="reel_id"  # Update if reel_id already exists
            ).execute()
            
            print(f"✅ Cached reel data in Supabase: {reel_id}")
            
            return True
            
        except ClientError as e:
            print(f"❌ R2 upload error: {e}")
            return False
        except Exception as e:
            print(f"❌ Error saving to Supabase + R2: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_video_url(self, reel_id: str) -> Optional[str]:
        """
        Get video URL (public or pre-signed)
        
        Returns:
            - If R2_PUBLIC_URL is set: public URL
            - Otherwise: pre-signed URL (valid for 1 hour)
        """
        try:
            # Get R2 key from metadata
            metadata = self.get_metadata(reel_id)
            if not metadata or not metadata.get("r2_video_key"):
                return None
            
            r2_key = metadata["r2_video_key"]
            
            # If public bucket URL is configured, return direct link
            if self.r2_public_url:
                return f"{self.r2_public_url}/{r2_key}"
            
            # Otherwise, generate pre-signed URL (valid for 1 hour)
            url = self.s3.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.r2_bucket,
                    'Key': r2_key
                },
                ExpiresIn=3600  # 1 hour
            )
            
            return url
            
        except Exception as e:
            print(f"❌ Error generating video URL: {e}")
            return None
    
    def delete_reel(self, reel_id: str) -> bool:
        """
        Delete reel from both Supabase and R2
        
        Args:
            reel_id: Reel to delete
        
        Returns:
            True if successful
        """
        try:
            # Get video key before deleting metadata
            metadata = self.get_metadata(reel_id)
            
            # Delete from Supabase
            self.supabase.table("reel_cache").delete().eq("reel_id", reel_id).execute()
            print(f"✅ Deleted from Supabase: {reel_id}")
            
            # Delete video from R2 if it exists
            if metadata and metadata.get("r2_video_key"):
                self.s3.delete_object(
                    Bucket=self.r2_bucket,
                    Key=metadata["r2_video_key"]
                )
                print(f"✅ Deleted from R2: {metadata['r2_video_key']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error deleting reel: {e}")
            return False
    
    def list_cached_reels(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List cached reels (for admin/debugging)
        
        Args:
            limit: Max number of reels to return
        
        Returns:
            List of reel metadata dicts
        """
        try:
            response = self.supabase.table("reel_cache").select("*").limit(limit).order("created_at", desc=True).execute()
            return response.data
        except Exception as e:
            print(f"❌ Error listing reels: {e}")
            return []


# ==================================================
# LOCAL FILESYSTEM IMPLEMENTATION (for development)
# ==================================================

class LocalFileStorage(StorageBackend):
    """
    Local filesystem storage (for testing without cloud services)
    Stores in ./cache/{reel_id}/
    """
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_reel_dir(self, reel_id: str) -> Path:
        """Get directory path for reel"""
        reel_dir = self.cache_dir / reel_id
        reel_dir.mkdir(exist_ok=True)
        return reel_dir
    
    def exists(self, reel_id: str) -> bool:
        metadata_path = self._get_reel_dir(reel_id) / "metadata.json"
        return metadata_path.exists()
    
    def get_metadata(self, reel_id: str) -> Optional[Dict[str, Any]]:
        metadata_path = self._get_reel_dir(reel_id) / "metadata.json"
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def save_reel_data(self, reel_id: str, metadata: Dict[str, Any], video_path: Optional[str] = None) -> bool:
        try:
            reel_dir = self._get_reel_dir(reel_id)
            
            # Save metadata
            with open(reel_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Copy video if provided
            if video_path and os.path.exists(video_path):
                import shutil
                shutil.copy(video_path, reel_dir / "video.mp4")
            
            print(f"✅ Cached reel data for {reel_id} locally")
            return True
            
        except Exception as e:
            print(f"❌ Error saving to local cache: {e}")
            return False
    
    def get_video_url(self, reel_id: str) -> Optional[str]:
        video_path = self._get_reel_dir(reel_id) / "video.mp4"
        return str(video_path) if video_path.exists() else None


# ==================================================
# STORAGE FACTORY
# ==================================================

def get_storage_backend(storage_type: str = None) -> StorageBackend:
    """
    Factory function to create storage backend
    
    Args:
        storage_type: "supabase_r2" or "local" (auto-detect from env if None)
    
    Returns:
        StorageBackend instance
    """
    if storage_type is None:
        storage_type = os.getenv("STORAGE_BACKEND", "local")
    
    if storage_type == "supabase_r2":
        return SupabaseR2Storage()
    
    elif storage_type == "local":
        cache_dir = os.getenv("LOCAL_CACHE_DIR", "./cache")
        return LocalFileStorage(cache_dir)
    
    else:
        raise ValueError(f"Unknown storage backend: {storage_type}. Choose: supabase_r2, local")


# ==================================================
# UTILITY FUNCTIONS
# ==================================================

def extract_reel_id_from_url(reel_url: str) -> str:
    """
    Extract reel ID from Instagram URL
    Examples:
        https://www.instagram.com/reel/ABC123xyz/ → ABC123xyz
        https://instagram.com/p/XYZ789/ → XYZ789
    """
    import re
    match = re.search(r'/(reel|p)/([A-Za-z0-9_-]+)', reel_url)
    if match:
        return match.group(2)
    
    # Fallback: hash the URL
    return hashlib.md5(reel_url.encode()).hexdigest()[:12]


def download_video(video_url: str, output_path: str) -> bool:
    """Download video from URL to local file"""
    try:
        response = requests.get(video_url, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Downloaded video to {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Video download failed: {e}")
        return False