############################## Libraries ####################################
# App
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from fpdf import FPDF
from tempfile import NamedTemporaryFile
##############################################################################


########################### Functions ########################################
# IBM Watson ML
from ibm_watson_machine_learning import APIClient
def ibm_model(api_key = 'wRfPsteh334TndKoAfmzPdGLGliGBa4q7YpDdMlvC9Ag',
              location = "eu-gb",
              deployment_uid = 'a8a0a600-0b59-417f-8f0d-1ad6590ad1ab',
              WML_SPACE_ID="d9bc5247-7acd-4601-85fd-910e2c49f299",
              fields = ['0','1','2'],
              values = [[1, 2, 3]]
              ):

    wml_credentials = {
        "apikey": api_key,
        "url": 'https://' + location + '.ml.cloud.ibm.com'
    }
    client = APIClient(wml_credentials)
    client.set.default_space(WML_SPACE_ID)
    scoring_payload = {"input_data": [{"fields": fields, "values": values}]}
    #scoring_payload
    predictions = client.deployments.score(deployment_uid, scoring_payload)
    return predictions

# Load data from input folder
@st.cache
def load_data(uploaded_file,type):
    if type == 'csv':
        return pd.read_csv(uploaded_file)
    if type == 'xlsx':
        return pd.read_excel(uploaded_file)
    else:
        print('Invalid format')


# Score record by input form
@st.cache
def score_record(input_values,model,predict_class = False, proba_treshold = 0.5):
    features = np.array(input_values).reshape(1, -1)
    # return class based on cut off
    if predict_class:
        scores = (model.predict_proba(features)[:, 1] >= proba_treshold).astype(int)[0]
    else:
        # return probabilities
        scores = model.predict_proba(features)[0]
    return scores


# Download function
@st.cache
def st_pandas_to_csv_download_link(_df:pd.DataFrame, file_name:str = "predictions.csv"):
    csv_exp = _df.to_csv(index=False)
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}" > Download Predictions (CSV) </a>'
    return href


@st.cache
def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'


import base64


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def donut_chart(input_values, model,label):
    scores = score_record(input_values, model)
    # Display prediction chart
    data = [scores[0], scores[1]]
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.figure(frameon=False, figsize=(5, 5))
    plt.gcf().gca().add_artist(centre_circle)
    plt.pie(data, labels=label, textprops={'color': "white"},
            colors=['skyblue', 'darkred'], autopct='%1.1f%%', pctdistance=0.85, )

    status = str(score_record(input_values, model, predict_class=True, proba_treshold=0.5))

    text = plt.annotate(status,
                         xy=(0, 0),
                         fontsize=80,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color='black')
    return fig



######################################################################


########################### Main function  ##########################

matrix_url = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fs-media-cache-ak0.pinimg.com%2Foriginals%2Fc5%2F9a%2Fd2%2Fc59ad2bd4ad2fbacd04017debc679ddb.gif&f=1&nofb=1"
cool_url = 'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwallpapercave.com%2Fwp%2Fwp6830287.gif&f=1&nofb=1'
cube_url = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ffree4kwallpapers.com%2Fuploads%2Foriginals%2F2015%2F10%2F27%2Fminimalist--wallpaper.jpg&f=1&nofb=1"
def main():
    st.markdown(
        """
        <style>
        .reportview-container {
            background: url("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwallpapercave.com%2Fwp%2Fwp6830287.gif&f=1&nofb=1");transparency:0.8;
            background-position: 100px 600px;
            background-size: cover
           
        }
       .sidebar .sidebar-content {
            background: url("url_goes_here")
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    #_________________________ Menu setting ________________________
    st.markdown('<center><h1>Welcome</h1></center>',unsafe_allow_html=True)
    st.markdown('<center><h1> This is my Machine Learning WebApp !</h1></center>', unsafe_allow_html=True)

    # Menu
    menu = ["Prediction","Bulk Prediction"]
    activity = st.selectbox("Activity", menu)
    # _________________________________________________________________

    # pickle model
    model = pickle.load(open('model', 'rb'))

    # ______________________ Prediction _____________________________
    if activity == "Prediction":

        submenu = ["Form", "Input text"]
        subactivity = st.selectbox("Choose how to upload the input", submenu)

        if subactivity == "Form":
            st.subheader("Fill the form and get the prediction")
            # Create form for predictions:

            with st.form(key="Input"):

                # Input of the form
                v1 = st.number_input(label="",step=1.,format="%.2f",key="v1")
                v2 = st.number_input(label="",step=1.,format="%.2f",key="v2")
                v3 = st.number_input(label="",step=1.,format="%.2f",key="v3")

                # Model User Inputs
                input_values = [v1, v2, v3]

                predict_button  = st.form_submit_button("Predict")

                if predict_button:
                    # call scoring API model IBM
                    # response = ibm_model(fields, values)["predictions"]
                    # prediction = response[0]["values"][0][0]
                    # pred_prob = response[0]["values"][0][1]

                    # ___________________ Get prediction _______________
                    label = ['NoRisk', 'Risk']
                    scores = score_record(input_values,model)
                    #Preview scores
                    st.text( scores )
                    # ___________________________________________________



                    st.header("Prediction")

                    #_________________Donut chart settings___________________________________

                    fig = donut_chart(input_values, model,label)

                    # Plot as image or canvas
                    # as image
                    # plt.axis('off')
                    # buf = BytesIO()
                    # fig.savefig(buf, format="png",transparent=True,frameon=False)
                    # st.image(buf)
                    # as plot
                    st.pyplot(fig)


                #____________________ Download Report ____________________________________
                    input_values = pd.DataFrame(input_values).T
                    # Edit
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font('Arial', 'B',25)

                    pdf.cell(w=195.0, h=0.0, align='C', txt="REPORT RESULTS", border=0)

                    with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                        fig.savefig(tmpfile.name,)

                        pdf.image(tmpfile.name, 10, 5)

                    pdf.set_font('Arial', '', 12)

                    html = create_download_link(pdf.output(dest="S").encode("latin-1"), "test")
                    st.markdown(html, unsafe_allow_html=True)
                    #____________________________________________________________________________

                    # ___________________________________________________


        if subactivity == "Input text":

                # Form input text
                with st.form(key="Input"):
                    text_input = st.text_input("Copy and paste data with a comma separator")
                    st.write("Example: 1 ,2.0 , 'A' ")

                    input_values = text_input.split(",")
                    predict_button = st.form_submit_button("Predict")

                if predict_button:
                    # call scoring API model IBM
                    # response = ibm_model(fields, values)["predictions"]
                    # prediction = response[0]["values"][0][0]
                    # pred_prob = response[0]["values"][0][1]

                    #___________________ Get prediction _______________
                    label = ['NoRisk', 'Risk']
                    scores = score_record(input_values,model)
                    #Preview scores
                    st.text( scores )

                    # Display prediction chart
                    label = ['NoRisk','Risk']
                    data = [scores[0],scores[1]]
                    #___________________________________________________

                    st.header("Prediction")

                    #_________________Donut chart settings ______________________________
                    fig = donut_chart(input_values, model, label)

                    # Plot as image or canvas
                    # as image
                    # plt.axis('off')
                    # buf = BytesIO()
                    # fig.savefig(buf, format="png",transparent=True,frameon=False)
                    # st.image(buf)
                    # as plot
                    st.pyplot(fig)
                #___________________________________________________

                    # ____________________ Download Report ____________________________________
                    input_values = pd.DataFrame(input_values).T
                    # Edit
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font('Arial', 'B', 25)

                    pdf.cell(w=195.0, h=0.0, align='C', txt="REPORT RESULTS", border=0)

                    with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                        fig.savefig(tmpfile.name, )

                        pdf.image(tmpfile.name, 10, 5)

                    pdf.set_font('Arial', '', 12)

                    html = create_download_link(pdf.output(dest="S").encode("latin-1"), "test")
                    st.markdown(html, unsafe_allow_html=True)
                    # ____________________________________________________________________________


    #_________________________________________________________________


    #______________________ Bulk Prediction _____________________________
    if activity == "Bulk Prediction":
        uploaded_file = st.sidebar.file_uploader(label='Upload the csv or excel file',
                                                 type=['csv','xlsx'])
        # Upload file
        global df
        if uploaded_file is not None:
            try:
                df = load_data(uploaded_file,'csv')
            except Exception as e :
                print(e)
                df = load_data(uploaded_file,'xlsx')

            st.write('Preview of file')
            st.write(df.head())

            st.write('Scoring')
            # Prediction with IBM model
            if st.button("Predict"):
                st.success('Bulk scoring successfully completed!')

                #define the features
                features = df

                # call scoring API model IBM

                # response = ibm_model(fields,values)["predictions"]
                # prediction = response[0]["values"][0][0]
                # pred_prob = response[0]["values"][0][1]
                scores = model.predict_proba(features)

                # rank the predictions
                ranked_risk = pd.DataFrame(scores,
                                           columns=['Probability negative','Probability positive']).sort_values(by = 'Probability positive'
                                          ,ascending=False)

                st.subheader('Ranked Precictions')
                st.write(ranked_risk)
                # Download results
                st.markdown(st_pandas_to_csv_download_link(ranked_risk, file_name="predictions.csv"), unsafe_allow_html=True)

                # Display highest
                st.header("Top 10 listed")
                top =ranked_risk.head(10)
                top = pd.concat([df.loc[top.index,:], top ],1)
                st.table(top)
    # _________________________________________________________________

######################################################################



######################### RUN web app #################################
if __name__ == '__main__':
        main()
########################################################################
