#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from PIL import Image
#from img_classification import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps
import numpy as np
import urllib.request
import zipfile

url = 'https://storage.googleapis.com/ml1000/mymodel.h5'
urllib.request.urlretrieve(url, 'mymodel.h5')
local = 'mymodel.h5'

def teachable_machine_classification(img, file):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = tf.keras.models.load_model(file)


    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = img

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (150, 150)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    #st.write(prediction[0])
    #return np.argmax(prediction) # return position of the highest probability
    return prediction



st.sidebar.markdown("Welcome - Demo")
#st.sidebar.selectbox("Link to the relevant datasets.", ["https://www.kaggle.com/vipoooool/new-plant-diseases-dataset",
#                                                        "https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection"])# kaggle links to dataset
page = st.sidebar.selectbox("Choose task", ["Image Classfication", "Covid Dashboard","Uber"])# pages


if page == "Image Classfication":
    st.title("Image Classification with TensorFlow 2.0")
    st.header("Dog Vs Cat")
    st.text("Upload a dog or cat image for classification")
# file upload and handling logic
    uploaded_file = st.file_uploader("Choose a dog or cat ...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Dog-Cat.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        #label = teachable_machine_classification(image, 'model/mymodel.h5')
        label = teachable_machine_classification(image, local)
        if label[0] >0.5:
            st.write("This is dog")
        else:
            st.write("This is cat")

elif page == "Covid Dashboard":
    import streamlit as st
    import pandas as pd
    import json
    import urllib.request
    import numpy as np
    import plotly.express as px
    import plotly.figure_factory as ff
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker


    def mpl_plot(data, label, is_log):
        st.markdown("### " + label)
        ax = plt.axes()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        plt.bar(dates, data, color="red", label=label)
        if (is_log == True):
            plt.yscale("log")
        plt.xticks(rotation=60, fontsize=5)
        st.pyplot()


    def overall_insights_window():
        st.markdown("## Overall Insights")
        # st.write("Total confirmed cases : %d \n\n Total recovered cases : %d \n\n Total deaths : %d" % (current_confirmed, current_recovered, current_deceased))
        st.line_chart(chart_data)

        st.markdown("### Closed cases stats")

        fig1 = plt.figure()
        ax1 = fig1.add_axes([0, 0, 1, 1])
        ax1.axis('equal')
        category = ['Recovered', 'Dead']
        number = [current_recovered, current_deceased]
        ax1.pie(number, labels=category, autopct='%1.2f%%', radius=0.5)
        st.pyplot()


    def detailed_charts_window():
        st.markdown("## Detailed Charts")
        daily_or_total = st.radio("Select what to plot", ('Daily data', 'Cumulative (total) data'))

        options = ["Confirmed", "Recovered", "Deaths"]
        data_type = st.selectbox("Select what to plot", options, index=0, key=None)

        if daily_or_total == "Daily data":
            if data_type == "Confirmed":
                mpl_plot(daily_confirmed, "Daily Confirmed", False)
            if data_type == "Recovered":
                mpl_plot(daily_recovered, "Daily Recovered", False)
            if data_type == "Deaths":
                mpl_plot(daily_deceased, "Daily Deaths", False)
        if daily_or_total == "Cumulative (total) data":
            log = st.checkbox("Logarithmic Scale", value=False)
            nature = "Linear"
            if (log == True):
                nature = "Logarithmic"
            if nature == "Linear":
                if data_type == "Confirmed":
                    mpl_plot(total_confirmed, "Total Confirmed", False)
                if data_type == "Recovered":
                    mpl_plot(total_recovered, "Total Recovered", False)
                if data_type == "Deaths":
                    mpl_plot(total_deceased, "Total Deaths", False)

            if nature == "Logarithmic":
                if data_type == "Confirmed":
                    mpl_plot(total_confirmed, "Total Confirmed", True)
                if data_type == "Recovered":
                    mpl_plot(total_recovered, "Total Recovered", True)
                if data_type == "Deaths":
                    mpl_plot(total_deceased, "Total Deaths", True)


    def statewise_data_window():
        st.markdown("## Statewise Data")

        state = st.selectbox("Select state", statelist, index=0, key=None)
        info_dict = {}
        info_dict = stateinfo[state]
        st.markdown("## %s" % (state))
        st.markdown("Total Confirmed Cases : %s" % (info_dict["confirmed"]))
        st.markdown("Total Active Cases : %s" % (info_dict["active"]))
        st.markdown("Total Deaths : %s" % (info_dict["deaths"]))
        st.markdown("Total Recovered : %s" % (info_dict["recovered"]))

        st.markdown("## Compare amongst states")
        state_data_active = []
        state_data_confirmed = []
        state_data_deaths = []
        state_data_recovered = []
        state_code_list = []
        for s in statelist:
            state_data_active.append(int(stateinfo[s]["active"]))
            state_data_confirmed.append(int(stateinfo[s]["confirmed"]))
            state_data_deaths.append(int(stateinfo[s]["deaths"]))
            state_data_recovered.append(int(stateinfo[s]["recovered"]))
            state_code_list.append(stateinfo[s]["statecode"])

        comparelist = ["Active Cases", "Confirmed Cases", "Deaths", "Recovered Cases"]
        compare = st.selectbox("What to compare?", comparelist, index=0, key=None)
        if compare == "Active Cases":
            plt.bar(state_code_list, state_data_active, color="blue", label=compare)
            plt.xticks(rotation=90, fontsize=7, fontweight="bold")
            st.pyplot()
        if compare == "Confirmed Cases":
            plt.bar(state_code_list, state_data_confirmed, color="blue", label=compare)
            plt.xticks(rotation=90, fontsize=7, fontweight="bold")
            st.pyplot()
        if compare == "Deaths":
            plt.bar(state_code_list, state_data_deaths, color="blue", label=compare)
            plt.xticks(rotation=90, fontsize=7, fontweight="bold")
            st.pyplot()
        if compare == "Recovered Cases":
            plt.bar(state_code_list, state_data_recovered, color="blue", label=compare)
            plt.xticks(rotation=90, fontsize=7, fontweight="bold")
            st.pyplot()


    st.title("COVID-19 India Dashboard")
    st.sidebar.title("Coronavirus India Dashboard")
    st.sidebar.markdown("India is one of the worst affected nations by the coronavirus outbreak.\
    	This tool aims at providing realtime insights on the outbreak in India in the form of interactive charts.")

    with urllib.request.urlopen("https://api.covid19india.org/data.json") as url:
        data = json.loads(url.read().decode())

    daily_confirmed = np.zeros(np.size(data["cases_time_series"]))
    daily_deceased = np.zeros(np.size(data["cases_time_series"]))
    daily_recovered = np.zeros(np.size(data["cases_time_series"]))

    total_confirmed = np.zeros(np.size(data["cases_time_series"]))
    total_recovered = np.zeros(np.size(data["cases_time_series"]))
    total_deceased = np.zeros(np.size(data["cases_time_series"]))

    dates = []

    i = 0
    for d in data["cases_time_series"]:
        daily_confirmed[i] = d["dailyconfirmed"]
        daily_deceased[i] = d["dailydeceased"]
        daily_recovered[i] = d["dailyrecovered"]
        total_confirmed[i] = d["totalconfirmed"]
        total_recovered[i] = d["totalrecovered"]
        total_deceased[i] = d["totaldeceased"]

        dates.append(d["date"])
        i = i + 1

    current_confirmed = total_confirmed[np.size(total_confirmed) - 1]
    current_recovered = total_recovered[np.size(total_recovered) - 1]
    current_deceased = total_deceased[np.size(total_deceased) - 1]

    chart_data = np.transpose([daily_confirmed, daily_recovered, daily_deceased])
    chart_data = pd.DataFrame(chart_data, columns=["Daily Confirmed Cases", "Daily Recovered Cases", "Daily Deaths"])

    statelist = []
    stateinfo = {}
    for d in data["statewise"]:
        if (d["state"] != "Total"):
            statelist.append(d["state"])
            stateinfo[d["state"]] = d

    st.sidebar.markdown("#### Total confirmed cases : %d " % (current_confirmed))
    st.sidebar.markdown("#### Total recovered cases : %d " % (current_recovered))
    st.sidebar.markdown("#### Total deaths : %d " % (current_deceased))

    st.sidebar.markdown("\n")
    window = st.sidebar.selectbox("Select:", ["Overall Insights", "Detailed Charts", "Statewise Data"], index=0,
                                  key=None)

    if (window == "Overall Insights"):
        overall_insights_window()
    if (window == "Detailed Charts"):
        detailed_charts_window()
    if (window == "Statewise Data"):
        statewise_data_window()
elif page == "Uber":
    import streamlit as st
    import pandas as pd
    import numpy as np
    import altair as alt
    import pydeck as pdk

    DATE_TIME = "date/time"
    DATA_URL = (
        "http://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz"
    )

    st.title("Uber Pickups in New York City")
    st.markdown(
        """
        This is a demo of a Streamlit app that shows the Uber pickups
        geographical distribution in New York City. Use the slider
        to pick a specific hour and look at how the charts change.
    
        [See source code](https://github.com/streamlit/demo-uber-nyc-pickups/blob/master/app.py)
        """)


    @st.cache(persist=True)
    def load_data(nrows):
        data = pd.read_csv(DATA_URL, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis="columns", inplace=True)
        data[DATE_TIME] = pd.to_datetime(data[DATE_TIME])
        return data


    data = load_data(100000)

    hour = st.slider("Hour to look at", 0, 23)

    data = data[data[DATE_TIME].dt.hour == hour]

    st.subheader("Geo data between %i:00 and %i:00" % (hour, (hour + 1) % 24))
    midpoint = (np.average(data["lat"]), np.average(data["lon"]))

    st.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={
            "latitude": midpoint[0],
            "longitude": midpoint[1],
            "zoom": 11,
            "pitch": 50,
        },
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=data,
                get_position=["lon", "lat"],
                radius=100,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
        ],
    ))

    st.subheader("Breakdown by minute between %i:00 and %i:00" % (hour, (hour + 1) % 24))
    filtered = data[
        (data[DATE_TIME].dt.hour >= hour) & (data[DATE_TIME].dt.hour < (hour + 1))
        ]
    hist = np.histogram(filtered[DATE_TIME].dt.minute, bins=60, range=(0, 60))[0]
    chart_data = pd.DataFrame({"minute": range(60), "pickups": hist})

    st.altair_chart(alt.Chart(chart_data)
        .mark_area(
        interpolate='step-after',
    ).encode(
        x=alt.X("minute:Q", scale=alt.Scale(nice=False)),
        y=alt.Y("pickups:Q"),
        tooltip=['minute', 'pickups']
    ), use_container_width=True)

    if st.checkbox("Show raw data", False):
        st.subheader("Raw data by minute between %i:00 and %i:00" % (hour, (hour + 1) % 24))
        st.write(data)

# In[ ]:

