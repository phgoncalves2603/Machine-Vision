from ultralytics import YOLO
from utils import read_videos, save_video
from trackers import tracker
import cv2
from team_assigner import TeamAssigner

def main():
    video_frames = read_videos("input_image/08fd33_4.mp4")
    t = tracker('models/best.pt')
    tracks = t.get_objects(video_frames, read_from_stub= True, stub_path='stubs/track_stubs.pki')
    
    team_assigner = TeamAssigner()
    team_assigner.assign(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['box'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # team_assigner = TeamAssigner()
    # team_assigner.assign(video_frames[0],tracks['players'][0])

    # for frame_num,player_track in enumerate(tracks['players']):
    #     for player_id, t in player_track.items():
    #         team = team_assigner.get_player_team(video_frames[frame_num],t['box'],player_id)
    #     tracks['players'][frame_num][player_id]['team'] = team
    #     tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # for id,player in tracks['players'][0].items():
    #     box = player["box"]
    #     frame = video_frames[0]

    #     cropped_image = frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])]

    #     cv2.imwrite(f'output_videos/cropped_img.jpg', cropped_image)
        

    
    output = t.draw(video_frames=video_frames, tracks=tracks)
    
    save_video(output, 'output_videos/output_video2.avi')


if __name__ =='__main__':
    main()
# model = YOLO('models/best.pt')

# results = model.predict("input_image/08fd33_4.mp4", save=True)


