import cv2

def read_videos(videos_path):
    cap=cv2.VideoCapture(videos_path)
    frames=[]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames
def save_video(frames, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frameSize = (frames[0].shape[1], frames[0].shape[0])  

    out = cv2.VideoWriter(output_path, fourcc, 24, frameSize)

    for frame in frames:
        out.write(frame)

    out.release()