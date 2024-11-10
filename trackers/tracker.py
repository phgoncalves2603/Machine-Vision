from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
import numpy as np
sys.path.append("../")# like an cd ../
from utils import get_box_center, get_box_width
class tracker:
    def __init__(self,model):
        self.model = YOLO(model)
        self.tracker = sv.ByteTrack()
        
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections+= detections_batch
        return detections
    
    def get_objects(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        tracks = {
            "players":[],
            "referees":[],
            "ball":[]
        }
        for i, detection in enumerate(detections):
            names = detection.names
            names_inv = {v:k for k,v in names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = names_inv["player"]
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            for frame_detection in detection_with_tracks:
                box = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id =frame_detection[4]
                if cls_id == names_inv['player']:
                    tracks['players'][i][track_id]={'box':box}
                if cls_id == names_inv["referee"]:
                    tracks['referees'][i][track_id]={'box':box}
            for frame_detection in detection_supervision:
                box = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == names_inv['ball']:
                    tracks['ball'][i][1] = {'box':box}
        
        if stub_path is not None:
            with open(stub_path,"wb") as f:
                pickle.dump(tracks, f)
        return tracks
    def draw_triangle(self,frame,box,color):
        y = int(box[1])
        x,_ = get_box_center(box)

        triangle = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours (frame, [triangle],0,color,cv2.FILLED)
        cv2.drawContours (frame, [triangle],0,(0,0,0),2)
        return frame
    def draw_ellipse(self, frame, box,color,track_id=None):
        y2=int(box[3])
        x_center,_ = get_box_center(box)
        width = get_box_width(box)

        cv2.ellipse(frame,center=(x_center,y2),axes=(int(width), int(0.35*width)), angle=0.0,color=color, thickness=1,lineType=cv2.LINE_4, startAngle=0,endAngle=360)
        
        #experimentation
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center - rectangle_width//2
        y1_rect = (y2-rectangle_height//2)+15
        y2_rect = (y2+rectangle_height//2)+15
        if track_id is not None:
            # cv2.rectangle(frame, 
            #               (x1_rect,y1_rect),
            #               (x2_rect,y2_rect),
            #               color,
            #               cv2.FILLED)
            x1_text = x1_rect+12 # padding
            if track_id>99:
                x1_text-=10
            
            cv2.putText(
                frame,
                str(track_id),
                (x1_text,y2_rect+15),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6, 
                (0,0,0),
                2
            )

        
        return frame

    def draw(self, video_frames, tracks):
        output = []
        for frame_num, frame in  enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            referee_dict = tracks['referees'][frame_num]

            #draw
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["box"],color, track_id)
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['box'],(0,255,255))
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['box'],(0,255,0))
            
            
            output.append(frame)
        
        return output


