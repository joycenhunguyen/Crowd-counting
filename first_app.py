import streamlit as st
from PIL import Image
import os
import engine

@st.cache
def load_image(image_file):
  img = Image.open(image_file)
  return img

#Save upload file
def save_uploadedfile(image_file, part_name):
  with open(os.path.join("part", part_name, image_file.name),"wb") as f:
    f.write(image_file.getbuffer())
  # return st.success("Saved File:{} to part_{}".format(image_file.name, part_name))
  
header = st.beta_container()

with header:
  st.title('Crowd Counting App')
  st.text('In this project, we use....')
def main():
  
  menu = ['Dense Image', 'Non-dense Image']
  choice = st.sidebar.selectbox('Options', options = menu)

  if choice == 'Dense Image':
    st.subheader (' High Density Crowd Image')
    image_file = st.file_uploader(' ', type= ['jpeg', 'png', 'jpg'], key = choice)
    if image_file is not None:
      st.image(load_image(image_file), width = 250)
      save_uploadedfile(image_file, 'A')
      st.write('Predicted number of people:', engine.run(image_file, 'A'))
                 
  else:
    st.subheader('Low Density Crowd Image')
    image_file_b = st.file_uploader(' ', type= ['jpeg', 'png', 'jpg'], key = choice)
    if image_file_b is not None:
      st.image(load_image(image_file_b), width = 250)
      save_uploadedfile(image_file, 'B')
      st.write(engine.run(image_file, 'B'))      

if __name__ == '__main__':
    main()
