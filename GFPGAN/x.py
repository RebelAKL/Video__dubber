import argparse
import cv2
import os
from gfpgan import GFPGANer
from codeformer import CodeFormer
from musetalk import process_video  

def upscale_frame(subframe, method):
    if method == "GFPGAN":
        # Load GFPGAN model
        gfpgan = GFPGANer(model_path="path_to_gfpgan_model.pth")
        _, _, enhanced_frame = gfpgan.enhance(subframe, has_aligned=True, only_center_face=True, paste_back=True)
        return enhanced_frame
    elif method == "CodeFormer":
        # Load CodeFormer model
        codeformer = CodeFormer(model_path="path_to_codeformer_model.pth")
        enhanced_frame = codeformer.enhance(subframe, fidelity=0.7)  # Fidelity can be adjusted
        return enhanced_frame
    else:
        raise ValueError("Invalid super-resolution method. Choose GFPGAN or CodeFormer.")

def enhance_video(input_video, input_audio, output_video, superres_method):
    # Process video with MuseTalk to get generated frames and regions
    processed_frames, generated_regions = process_video(input_video, input_audio)
    
    enhanced_frames = []
    for frame, region in zip(processed_frames, generated_regions):
        x, y, w, h = region  # Region of the generated subframe
        subframe = frame[y:y+h, x:x+w]
        original_res = frame.shape[:2]
        generated_res = subframe.shape[:2]
        
        # Check resolution ratio
        if generated_res[0] < original_res[0] or generated_res[1] < original_res[1]:
            ratio = max(original_res[0] / generated_res[0], original_res[1] / generated_res[1])
            subframe = upscale_frame(subframe, superres_method)
        
        # Replace the subframe in the original frame
        frame[y:y+h, x:x+w] = subframe
        enhanced_frames.append(frame)
    
    # Combine enhanced frames into a video
    output_path = os.path.join(os.path.dirname(output_video), "enhanced_temp.mp4")
    frame_height, frame_width = enhanced_frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
    for frame in enhanced_frames:
        out.write(frame)
    out.release()
    
    # Combine video with audio
    os.system(f"ffmpeg -i {output_path} -i {input_audio} -c:v copy -c:a aac {output_video}")
    os.remove(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance video quality using super-resolution.")
    parser.add_argument("--superres", type=str, choices=["GFPGAN", "CodeFormer"], required=True, help="Super-resolution method to use.")
    parser.add_argument("-iv", "--input_video", type=str, required=True, help="Input video file.")
    parser.add_argument("-ia", "--input_audio", type=str, required=True, help="Input audio file.")
    parser.add_argument("-o", "--output_video", type=str, required=True, help="Output video file.")
    
    args = parser.parse_args()
    
    enhance_video(args.input_video, args.input_audio, args.output_video, args.superres)
