from modules.utils import *
from modules.pose_detection import *

# title and layouting
st.sidebar.title('TZ Simple Bike Fitting')
st.sidebar.caption("Prototype")
st.sidebar.write('Upload your image')
col1, col2, col3 = st.columns(3)

url = st.sidebar.text_input('The URL link')

file_bytes = read_file_from_url(url)
image = Image.open(BytesIO(file_bytes))

st.image(image, caption="Sunrise by the mountains", use_column_width=True)

# pose detection core

# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index#models

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='./src/pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

mp_pose = mp.solutions.pose

# STEP 3: Load the input image.

img_url = image

# STEP 4: Detect pose landmarks from the input image.

fit = bike_fit_mod(img_url,detector,mp_pose)

# STEP 5: Process the detection result. In this case, visualize it.
st.write('Elbow angle: {} degree'.format(fit['elbow_angle']))
st.write('Knee angle: {} degree'.format(fit['knee_angle']))
st.image(fit['annotated_image'], channels="RGB", use_column_width=True)
