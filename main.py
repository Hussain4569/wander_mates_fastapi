import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deepface import DeepFace
import cv2
import os
import requests
import uuid
import uvicorn
from datetime import datetime

app = FastAPI()

class FaceComparisonRequest(BaseModel):
    video_path: str
    image_path: str

def download_file(url, suffix):
    """Download a file from a URL and return the local temporary path"""
    temp_dir = tempfile.gettempdir()  # Get the system's temporary directory
    local_filename = os.path.join(temp_dir, f"{uuid.uuid4()}{suffix}")

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        return local_filename
    else:
        print(f"exception in downloading {response.text}")
        raise Exception(f"Failed to download file: {url}")


def extract_faces_from_video(video_path):
    """Extract frames from video and return a list of local frame paths"""
    video_capture = cv2.VideoCapture(video_path)
    frame_skip = 10  # Process every 10th frame for efficiency
    frame_count = 0
    frame_paths = []
    while True:
        ret, frame = video_capture.read()
        temp_dir = tempfile.gettempdir()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            frame_path = os.path.join(temp_dir, f"temp_frame_{uuid.uuid4()}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)

        frame_count += 1

    video_capture.release()
    return frame_paths

@app.post("/compare_faces/")
def compare_faces(request: FaceComparisonRequest):
    """Compare face in image with faces in video using DeepFace"""

    try:
        video_file = download_file(request.video_path, ".mp4")
        image_file = download_file(request.image_path, ".jpg")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    frame_paths = extract_faces_from_video(video_file)
    if not frame_paths:
        raise HTTPException(status_code=400, detail="No frames extracted from video")

    result = None
    matchesFound = 0

    for frame_path in frame_paths:
        try:
            result = DeepFace.verify(img1_path=image_file, img2_path=frame_path, enforce_detection=False)
            if result["verified"]:
                matchesFound += 1
            if matchesFound >= len(frame_paths) / 2:
                break
        except Exception as e:
            print(f"Error processing frame: {frame_path}: {e}")
            continue

    # Clean up all temporary files
    for f in frame_paths + [video_file, image_file]:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Failed to delete temp file {f}: {e}")

    if matchesFound > len(frame_paths) / 2:
        return { "match": True, "matchesFound": matchesFound, "message": "Face in image matches a face in video", "result": result }
    else:
        return { "match": False, "matchesFound": matchesFound, "message": "No match found", "result": result }


if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)