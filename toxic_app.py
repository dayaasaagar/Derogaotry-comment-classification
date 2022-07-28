# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 21:04:07 2022

@author: dayaaasaagar
"""

import streamlit as st 
import lstmword 
from PIL import Image
from bert_toxic import bertktrain
image = Image.open('2toxic.jpg')

st.image(image, caption='123')

st.title("Derogatory Comment detector")
#st.sidebar.title("select the method")
#st.sidebar.selectbox("Please select", ["algo1","algo2","alogo3"])


#sentence=st.text_input("input your  sentence")
#if sentence:
#    lstmword.toxicity_level(sentence)
    
           
option=["lstm","bert"]
choice = st.sidebar.selectbox("Select  your choice", options=option)
if choice=="lstm":
    sentence=st.text_input("input your  sentence")
    if sentence:
        lstmword.toxicity_level(sentence)
if choice =="bert":
    sentence=st.text_input("enter your sentence ")
    if sentence:
        bertktrain(sentence)
    
