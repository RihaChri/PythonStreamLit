https://www.youtube.com/watch?v=B0MUXtmSpiA 
import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Define header")
header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()


#css code here
st.markdown(
    """
    <style>
    .main {
    background-color: #FA6B6D;
    }
    </style> 
    """,
    unsafe_allow_html=True
)

@st.cache
def get_data(filename):
    return pd.read_csv(filename, sep=',')


with header:
    st.title("Crocs!")
    st.text('This is some text')
    
with dataset:
    st.header('Alligator')
    st.text('This is some text')
    
    data = get_data('yellow_tripdata_2021-01.csv')
    st.write(data.head())
    
    st.subheader('Pick-up location ID')
    pulocation_dist = pd.DataFrame(data['PULocationID'].value_counts()).head(50)
    st.bar_chart(pulocation_dist)
    
    
    
with features:
    st.header('Caiman')
    st.text('This is some text')
    
    st.markdown('* **first feature:** I created this feature...dsd')
    st.markdown('* **second feature:** I created this feature ...ds')
    
with modelTraining:
    st.header('Crocodile')
    st.text('This is some text')
    
    sel_col, disp_col = st.columns(2)
    max_depth = sel_col.slider("hi blabla", min_value=10, max_value=100,value=20,step=10)
    
    n_estimators = sel_col.selectbox("How many trees should be there be?", options=[100,200,300,"no limit"],index=0)
    
    input_feature = sel_col.text_input("hich feature should be used as input feature","PULocationID")
    
    regr=RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
    X=data[[input_feature]]
    Y=data[['trip_distance']]
    
    #regr.fit(X,Y)
    #prediction=regr.predict(Y)
    #disp_col.subheader('Mean abs error: ')
    #disp_col.write(mean_absolute_error(Y,prediction))
    
    #disp_col.subheader('Mean squared error of the model: ')
    #disp_col.write(mean_squared_error(Y,prediction))
    
    #disp_col.subheader('R squared score of the model:')
    #disp_col.write(r2_score(Y,prediction))
    
    disp_col.text('here are the features of the dataset')
    disp_col.table(data.columns)