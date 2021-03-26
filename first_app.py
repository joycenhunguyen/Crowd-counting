import streamlit as st
from PIL import Image
import os
import engine
import numpy as np


@st.cache
def load_image(image_file):
  img = Image.open(image_file)
  return img

# Set page config
apptitle = 'Crowd Counting'

st.set_page_config(page_title=apptitle, page_icon=":fox_face:")

#Save upload file
def save_uploadedfile(image_file, part_name):
  with open(os.path.join("part", part_name, image_file.name),"wb") as f:
    f.write(image_file.getbuffer())
  # return st.success("Saved File:{} to part_{}".format(image_file.name, part_name))
  
header = st.beta_container()

with header:
  st.title('Crowd Counting App')
  st.text('In this project, we use....')
#Sidebar

st.sidebar.title('Select your image')


def main():
  
  menu = ['Dense Image', 'Non-dense Image']
  choice = st.sidebar.selectbox('Options', options = menu)

  if choice == 'Dense Image':
    st.subheader (' High Density Crowd Image')
    image_file = st.sidebar.file_uploader(' ', type= ['jpeg', 'png', 'jpg'], key = choice)
    
    if image_file is not None:
      st.image(load_image(image_file), width = 500, caption = 'Uploaded image')
      save_uploadedfile(image_file, 'A')
      image_heat=engine.run(image_file,'A')
      st.image(image_heat,width = 500, caption = 'heatmap image')
      st.write('Predicted number of people:', round(np.sum(image_heat)))
                 
  else:
    st.subheader('Low Density Crowd Image')
    image_file = st.sidebar.file_uploader(' ', type= ['jpeg', 'png', 'jpg'], key = choice)
    if image_file is not None:
      st.image(load_image(image_file), width = 500)
      save_uploadedfile(image_file, 'B')
      image_heat=engine.run(image_file,'B')
      st.image(image_heat,width = 500, caption = 'heatmap image')
      st.write('Predicted number of people:', round(np.sum(image_heat)))    

if __name__ == '__main__':
    main()


st.subheader("About this app")
st.markdown("""
This app displays data....
""")
