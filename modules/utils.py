import streamlit as st
import pandas as pd
from io import BytesIO
import requests
from PIL import Image
import streamlit as st


# https://github.com/streamlit/streamlit/blob/c74227ac25b35b06f13216bdbf902cb7e636ee42/docs/api-examples-source/charts.image.py#L17
@st.cache_data
def read_file_from_url(url):
    return requests.get(url).content
