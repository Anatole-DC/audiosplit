from streamlit import write, file_uploader

write("# Welcome to audiosplit")

file_uploader("Upload a music", type=["mp3", "wav"])
