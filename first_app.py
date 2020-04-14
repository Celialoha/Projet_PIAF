import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os, urllib
import time
import joblib
from keras.models import load_model

cni = load_model('C:/Users/Celia LAGARDE RAMORA/Desktop/CNI_TL_ImageNet.h5')
 
def cni_or_not(image,cni):        
    x = image.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x = x/255.
    #x = preprocess_input(x, mode='caffe')
    preds = cni.predict(x)
    return preds    

st.title('Authentification de CNI')


uploaded_file = st.file_uploader("Téléchargez votre carte d'identité", type=['png','jpg'])
'Veuillez patienter, nous étudions votre photo...'

if uploaded_file is not None:
    st.image(uploaded_file)
    resultat = cni_or_not(uploaded_file)
    st.write(resultat)

## Add a placeholder
#latest_iteration = st.empty()
#bar = st.progress(0)
#
#for i in range(10):
#  # Update the progress bar with each iteration.
#  latest_iteration.text(f'Chargement {i+1}')
#  bar.progress(i + 1)
#  time.sleep(0.1)
#
#'... Voici ce que nous en pensons !'


