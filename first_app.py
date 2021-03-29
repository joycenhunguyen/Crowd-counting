import streamlit as st
from PIL import Image
import os
import engine
import numpy as np
import matplotlib.pyplot as plt

@st.cache
def load_image(image_file):
  img = Image.open(image_file)
  return img

# Set page config
apptitle = 'Crowd Counting'

st.set_page_config(page_title=apptitle, page_icon=":fox_face:",layout='wide')

#Save upload file
def save_uploadedfile(image_file, part_name):
  with open(os.path.join("part", part_name, image_file.name),"wb") as f:
    f.write(image_file.getbuffer())
  # return st.success("Saved File:{} to part_{}".format(image_file.name, part_name))
  
header = st.beta_container()

with header:
  st.markdown(
    """<style>     .container {display: flex;flex-wrap:wrap;}</style>""",unsafe_allow_html=True)
  st.markdown("""<h1 style='text-align:center;background-color:#003399;color:white;padding:20px'>Crowd Counting App</h1>
  <hr>
  <h2 style='color:#003399'><b>Instruction </b></h2>
  <p>Below are examples of bad images (not a crowd), Dense vs Non-dense imanges for your references.</p>
  <p class='container'>
  <span>
  <img src="https://bambiniphoto.sg/wp-content/uploads/family-photography-bambini-025.jpg" alt="avoid1" style="width:300px;height:150px;padding:5px">
  <figcaption style='text-align:center;color:red'>Not good</figcaption>
  </span>
  <span>
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRj42KnQQ3y_upUSOSYC_dC33tSUxsj1ck2iQ&usqp=CAU" alt="avoid2" style="width:200px;height:150px;padding:5px">
  <figcaption style='text-align:center;color:red'>Not good</figcaption>
  </span>
  <span>
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR4PqTHWIxkQnoWIwp4jHhizRRzNRuhaKQS-MsEL5P82OUbfam2sLcURc_WvsNYk1vlGs8&usqp=CAU" alt="avoid1" style="width:200px;height:150px;padding:5px">
  <figcaption style='text-align:center;color:green'>Dense</figcaption>
  </span>
  <span>
  <img src="https://ak.picdn.net/shutterstock/videos/4974800/thumb/1.jpg" alt="avoid2" style="width:200px;height:150px;padding:5px">
  <figcaption style='text-align:center;color:green'>Dense</figcaption>
  </span>
    <span>
  <img src="https://as.cornell.edu/sites/default/files/styles/4_5/public/field/image/crowd450.jpg?itok=ppSyCQT7" alt="good1" style="width:200px;height:150px;padding:5px">
  <figcaption style='text-align:center;color:blue'>Non-dense</figcaption>
  </span>
  <span>
  <img src="http://marybaldwin.edu/wp-content/uploads/2018/11/MDCHS-1024x343.jpg" alt="good2" style="width:200px;height:150px;padding:5px">
  <figcaption style='text-align:center;color:blue'>Non-dense</figcaption>
  </span>
  </p>
  <hr>
  """, unsafe_allow_html=True)
#Sidebar

st.sidebar.title('Select your image')


def main():
  
  menu = ['Dense Image', 'Non-dense Image']
  choice = st.sidebar.selectbox('Options', options = menu)

  if choice == 'Dense Image':
    st.markdown("""<h2 style='color:#003399'><b>Result</b></h2>""",unsafe_allow_html=True)
    image_file = st.sidebar.file_uploader(' ', type= ['jpeg', 'png', 'jpg'], key = choice)
    
    if image_file is not None:
      col1, col2= st.beta_columns(2)
      with col1:
        st.image(load_image(image_file), use_column_width='auto', caption = 'Uploaded image')
        save_uploadedfile(image_file, 'A')
      with col2:
        image_heat=engine.run(image_file,'A')  
        fig=plt.figure(figsize=(3,2))
        plt.imshow(image_heat, cmap=plt.cm.jet)
        plt.axis('off')
        st.pyplot(fig)
        st.write('Predicted number of people:', round(np.sum(image_heat)))
  
                 
  else:
    st.markdown("""<h2 style='color:#003399'><b>Result</b></h2>""",unsafe_allow_html=True)
    image_file = st.sidebar.file_uploader(' ', type= ['jpeg', 'png', 'jpg'], key = choice)
    if image_file is not None:
      col1, col2 = st.beta_columns(2)
      with col1:
        st.image(load_image(image_file), use_column_width='auto', caption = 'Uploaded image')
        save_uploadedfile(image_file, 'B')
      with col2:
        image_heat=engine.run(image_file,'B')
        fig=plt.figure(figsize=(3,2))
        plt.imshow(image_heat, cmap=plt.cm.jet)
        plt.axis('off')
        st.pyplot(fig)
        st.write('Predicted number of people:', round(np.sum(image_heat)))
 

if __name__ == '__main__':
    main()


# st.subheader("---")
# st.markdown("""<p style='text-align:center'>
# Made by Nhu Nguyen, Thuong Nguyen, Radim Musalek. Deep Learning model is pretained from https://github.com/ZhengPeng7/CSRNet-Keras.</p>
# """,unsafe_allow_html=True)
