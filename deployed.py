
import numpy as np 
import pandas as pd
import streamlit as st
import joblib

# Lets load all the instances required over here 
with open('transformer.joblib','rb') as file:
    transformer = joblib.load(file)

# lets load the model
with open ('final_model.joblib','rb') as file:
    model = joblib.load(file)

st.title('INN HOTEL GROUP')
st.header(':blue[This application will predict the chances of booking cancellation]')


# lets take input from user 

amnth = st.slider('Select Your month or arrival',min_value=1,max_value=12,step=1)
wkd_lambda = (lambda x: 0 if x=='Mon' else
              1 if x=='Tue' else
              2 if x=='Wed' else
              3 if x=='Thurs' else
              4 if x=='Fri' else
              5 if x=='Sat' else 6)
awkd = wkd_lambda(st.selectbox('Select your weekday of arrival',['--','Mon','Tues','Wed','Thurs','Fri','Sat','Sun']))
dwkd = awkd = st.selectbox('Select your weekday of departure',['--','Mon','Tues','Wed','Thurs','Fri','Sat','Sun'])
wkend = st.number_input('Enter How many weekend nights are there in stay',min_value=0)
wk = st.number_input('Enter How many week nights are there in stay',min_value=0)
totn = wkend + wk
mkt = (lambda x: 0 if x=='Offline' else 1)(st.selectbox('How the booking has been made',['--','Offline','Online']))
lt = st.number_input('How many days prior the booking was made',min_value=0)
price = st.number_input('What is the avg price per room',min_value=0)
adults = st.number_input('How many adult members in booking',min_value=0)
spcl = st.selectbox('Select the number of special request made',[0,1,2,3,4,5])
park = (lambda x: 0 if x=='No' else 1)(st.selectbox('Does guest need parking spece',['--','Yes','No']))

# transform the data
lt_t,price_t = transformer.transform([[lt,price]])[0]

# create the input list 
input_list = [lt_t,spcl,price_t,adults,wkend,park,wk,mkt,amnth,totn,dwkd]

# Make prediction
prediction = model.predict_proba([input_list])[:,1][0]

# Lets show the probability 
if st.button('predict'):
    st.success(f'Cancellation chances: {round(prediction,4)*100}%')
