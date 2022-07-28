# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 19:09:05 2022

@author: dayaaasaagar
"""

import ktrain
from ktrain import text
import streamlit as st 
def bertktrain(sentence):
    p = ktrain.load_predictor('bertpredictor')
    result=p.predict(sentence)
    st.write('Toxic:         {:.0%}'.format(result[0][1]))
    st.write('Severe Toxic:  {:.0%}'.format(result[1][1]))
    st.write('Obscene:       {:.0%}'.format(result[2][1]))
    st.write('Threat:        {:.0%}'.format(result[3][1]))
    st.write('Insult:        {:.0%}'.format(result[4][1]))
    st.write('Identity Hate: {:.0%}'.format(result[5][1]))
    