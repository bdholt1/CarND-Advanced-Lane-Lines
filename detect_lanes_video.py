
import cv2
from moviepy.editor import VideoFileClip
import lane
import pickle


detector = lane.LaneDetector("calibration.p", debug=False)

output = 'project_video_result.mp4'
clip = VideoFileClip('project_video.mp4')
project_clip = clip.fl_image(detector.process)

project_clip.write_videofile(output, audio=False)
