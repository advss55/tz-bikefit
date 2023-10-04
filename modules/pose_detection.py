import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import gradio as gr
from PIL import Image
from io import BytesIO
import requests

# calculate angle between landmark coordinates
# https://github.com/Pradnya1208/Squats-angle-detection-using-OpenCV-and-mediapipe_v1/blob/main/Squat%20pose%20estimation.ipynb
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle = 360-angle

    return angle

# draw landmarks on image
def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # idx_viz = [12, 14, 16, 24, 26, 28] # TODO: draw only these landmarks

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks # TODO: draw only important landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())

  return annotated_image

# bike fit
def bike_fit_mod(input_img,detector,mp_pose):
  # Load the input image.
  # image = mp.Image.create_from_file(input_img)
  image = mp.Image(
      image_format=mp.ImageFormat.SRGB, data=np.asarray(input_img))

  # Detect pose landmarks from the input image.
  detection_result = detector.detect(image)

  # prepare dict containing the fit
  fit = {}

  # get landmark
  landmarks = detection_result.pose_landmarks[0]

  shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
  elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
  wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
  hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
  knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
  ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

  shoulder_coordinate = [shoulder.x, shoulder.y]
  elbow_coordinate = [elbow.x, elbow.y]
  wrist_coordinate = [wrist.x, wrist.y]
  hip_coordinate = [hip.x, hip.y]
  knee_coordinate = [knee.x, knee.y]
  ankle_coordinate = [ankle.x, ankle.y]

  fit['elbow_angle'] = calculate_angle(shoulder_coordinate, elbow_coordinate, wrist_coordinate)
  fit['knee_angle'] = calculate_angle(hip_coordinate, knee_coordinate, ankle_coordinate)
  fit['annotated_image'] = draw_landmarks_on_image(image.numpy_view(), detection_result)

  return fit

