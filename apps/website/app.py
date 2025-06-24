from base64 import b64encode, b64decode

from streamlit import write, file_uploader, audio
from apps.website.components import multitrack

write("# Welcome to audiosplit")

uploaded_file = file_uploader("Upload a music", type=["mp3", "wav"])
if uploaded_file:
    encoded_audio_file = b64encode(uploaded_file.read()).decode("ascii")

    multitrack(encoded_audio_file)

# write(return_value)
