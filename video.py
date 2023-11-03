import streamlit as st
from moviepy.editor import VideoFileClip
from part3 import pipeline  

video_output = 'project_video_output_test_8.mp4'

uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_file is not None:
    video_bytes = uploaded_file.read()

    with open('temp_video.mp4', 'wb') as f:
        f.write(video_bytes)

    clip1 = VideoFileClip('temp_video.mp4')
    output_fps = 25
    

    video_clip = clip1.fl_image(pipeline)  
    video_clip.write_videofile(video_output)
    

    st.video(video_output)