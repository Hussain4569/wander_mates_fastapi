from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deepface import DeepFace
import cv2
import os

app = FastAPI()

class FaceComparisonRequest(BaseModel):
    video_path: str
    image_path: str

def extract_faces_from_video(video_path):
    """Extract frames from video and return a list of frame paths"""
    video_capture = cv2.VideoCapture(video_path)
    frame_skip = 10  # Process every 10th frame for efficiency
    frame_count = 0
    frame_paths = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            frame_path = f"frames/temp_frame_{frame_count}.jpg"
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)

        frame_count += 1

    video_capture.release()
    return frame_paths

@app.post("/compare_faces/")
def compare_faces(request: FaceComparisonRequest):
    """Compare face in image with faces in video using DeepFace"""
    print(f"Comparing image: {request.image_path} with video: {request.video_path}")
    frame_paths = extract_faces_from_video("../" + request.video_path)
    if not frame_paths:
        raise HTTPException(status_code=400, detail="No frames extracted from video")

    result = None
    matchesFound = 0
    for frame_path in frame_paths:
        try:
            #print(f"Comparing image with frame: {frame_path}")
            result = DeepFace.verify(img1_path="../" + request.image_path, img2_path=frame_path, enforce_detection=False)
            #print(result)
            if result["verified"]:
                matchesFound += 1
        
        except Exception as e:
            print(f"Error processing frame: {frame_path}")
            print(e)
            continue
    
    #empty the frames directory
    for frame in frame_paths:
        os.remove(frame)
    
    if matchesFound > (frame_paths.__len__() / 2): #if more than half of the frames have a match
        return {"match": True, "matchesFound": matchesFound, "message": "Face in image matches a face in video", "result": result}
    else:
        return {"match": False, "matchesFound": matchesFound, "message": "No match found", "result": result}
