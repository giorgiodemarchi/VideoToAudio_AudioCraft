
## TODO: add to requirements
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from torchvision.io import read_video
#####

import logging 
import torch 
import typing as tp
import random
from dataclasses import dataclass

from .audio_dataset import BaseInfo, SegmentInfo, AudioDataset
from .audio_utils import convert_audio

logger = logging.getLogger(__name__)

@dataclass(order=True)
class AudioVideoMeta(BaseInfo):
    """
    Handles metadata
    """
    path: str
    video_path: str  # path to the video file
    duration: float
    sample_rate: int
    amplitude: tp.Optional[float] = None
    weight: tp.Optional[float] = None

    @classmethod
    def from_dict(cls, dictionary: dict):
        base = super().from_dict(dictionary)
        base['video_path'] = dictionary.get('video_path', '')
        return cls(**base)

class AudioVideoDataset(AudioDataset):
    """
    Audio-Video dataset, inhereting from AudioDataset
    
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s3_client = boto3.client('s3')   ### TODO: Add credentials
        self.bucket_name = 'adorno-audioset'

    def _download_from_s3(self, s3_path):
        """Helper function to download media files from S3"""
        try:
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_path)
            return obj['Body'].read()
        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error(f"Credentials not available for AWS S3. {e}")
            return None

    def _load_video(self, video_path, seek_time, duration):
        """Load a segment of a video given the path, seek time, and duration."""
        video_bytes = self._download_from_s3(video_path)
        video, _, _ = read_video(video_bytes, start_pts=seek_time, end_pts=seek_time+duration)
        return video

    def __getitem__(self, index):
        """Overrides the __getitem__ to load both audio and video."""
        if self.segment_duration is None:
            media_meta = self.meta[index]
            audio, sr = self._audio_read(media_meta.path)
            video = self._load_video(media_meta.video_path, 0, media_meta.duration)
        else:
            rng = torch.Generator().manual_seed(self.shuffle_seed + index)
            media_meta = self.sample_file(index, rng)
            seek_time = random.uniform(0, media_meta.duration - self.segment_duration)
            audio, sr = self._audio_read(media_meta.path, seek_time, self.segment_duration)
            video = self._load_video(media_meta.video_path, seek_time, self.segment_duration)
        
        audio = convert_audio(audio, sr, self.sample_rate, self.channels)
        
        if self.return_info:
            segment_info = SegmentInfo(media_meta, seek_time, audio.shape[-1], audio.shape[-1], self.sample_rate, audio.shape[0])
            return audio, video, segment_info
        else:
            return audio, video

