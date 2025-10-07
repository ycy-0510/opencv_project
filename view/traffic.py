import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
import numpy as np
import av

st.title("Traffic Live Stream")

st.image("http://127.0.0.1:8000/stream")