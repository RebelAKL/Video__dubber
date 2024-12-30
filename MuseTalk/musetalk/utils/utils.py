import os
import cv2
import numpy as np
import torch
from gfpgan import GFPGANer
from codeformer import CodeFormer

# Ensure FFMPEG is configured
ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None:
    print("Please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static")
elif ffmpeg_path not in os.getenv('PATH'):
    print("Adding ffmpeg to PATH")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"

# Import MuseTalk modules
from musetalk.whisper.audio2feature import Audio2Feature
from musetalk.models.vae import VAE
from musetalk.models.unet import UNet, PositionalEncoding

def load_all_model():
    audio_processor = Audio2Feature(model_path="./models/whisper/tiny.pt")
    vae = VAE(model_path="./models/sd-vae-ft-mse/")
    unet = UNet(unet_config="./models/musetalk/musetalk.json",
                model_path="./models/musetalk/pytorch_model.bin")
    pe = PositionalEncoding(d_model=384)
    return audio_processor, vae, unet, pe

def get_file_type(video_path):
    _, ext = os.path.splitext(video_path)
    if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        return 'image'
    elif ext.lower() in ['.avi', '.mp4', '.mov', '.flv', '.mkv']:
        return 'video'
    else:
        return 'unsupported'

def get_video_fps(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

def apply_super_resolution(image, method="GFPGAN"):
    """
    Apply super-resolution to an image using GFPGAN or CodeFormer.
    Args:
        image (numpy.ndarray): The image to enhance.
        method (str): The super-resolution method to use ("GFPGAN" or "CodeFormer").
    Returns:
        numpy.ndarray: The enhanced image.
    """
    if method == "GFPGAN":
        gfpgan = GFPGANer(model_path="path_to_gfpgan_model.pth")
        _, _, enhanced_image = gfpgan.enhance(image, has_aligned=True, only_center_face=True, paste_back=True)
        return enhanced_image
    elif method == "CodeFormer":
        codeformer = CodeFormer(model_path="path_to_codeformer_model.pth")
        enhanced_image = codeformer.enhance(image, fidelity=0.7)
        return enhanced_image
    else:
        raise ValueError(f"Invalid super-resolution method: {method}")

def datagen(whisper_chunks, vae_encode_latents, batch_size=8, delay_frame=0):
    whisper_batch, latent_batch = [], []
    for i, w in enumerate(whisper_chunks):
        idx = (i + delay_frame) % len(vae_encode_latents)
        latent = vae_encode_latents[idx]
        whisper_batch.append(w)
        latent_batch.append(latent)

        if len(latent_batch) >= batch_size:
            whisper_batch = np.stack(whisper_batch)
            latent_batch = torch.cat(latent_batch, dim=0)
            yield whisper_batch, latent_batch
            whisper_batch, latent_batch = [], []

    # Handle the last batch (may be smaller than batch size)
    if len(latent_batch) > 0:
        whisper_batch = np.stack(whisper_batch)
        latent_batch = torch.cat(latent_batch, dim=0)
        yield whisper_batch, latent_batch
