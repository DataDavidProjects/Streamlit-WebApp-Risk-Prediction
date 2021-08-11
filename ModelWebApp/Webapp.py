############################## Libraries ####################################
# App
import time

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from fpdf import FPDF
from tempfile import NamedTemporaryFile
import sklearn

np.set_printoptions(precision = 4, suppress = True)
##############################################################################
# Heroku link:  https://ensemblemlcorporates.herokuapp.com/

########################### Functions ########################################
# IBM Watson ML
from ibm_watson_machine_learning import APIClient
def ibm_model(
        fields,
        values,
        api_key = 'wRfPsteh334TndKoAfmzPdGLGliGBa4q7YpDdMlvC9Ag',
        location = "eu-gb",
        deployment_uid = '9b3a85e0-76cf-4f7b-82c4-272464237280',
        WML_SPACE_ID="d9bc5247-7acd-4601-85fd-910e2c49f299",
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


# __________________________ helper function ______________________________
def data_cleaning(df):
    # Drop columns with missing rate > n
    n = 0.4
    missing_perc = df.isna().sum().sort_values(ascending=False) / df.shape[0]
    imputable_columns = missing_perc[missing_perc < n].keys()
    df = df.loc[:, imputable_columns]

    return df


# Pipelines Class custom
from sklearn.base import BaseEstimator, TransformerMixin


# Count the number of missing values per row
class missing_counter(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        missing_collection = np.isnan(X).sum(axis=1)

        return np.c_[X, missing_collection]


# Count the number of outliers per row based on 2 z scores
class outlier_flag_counter(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Scale with standard values
        z_scaler = sklearn.preprocessing.StandardScaler()
        # select just features for scaling
        zratios = pd.DataFrame(z_scaler.fit_transform(X))
        # Flag outliers counter
        outlier_flag_counter = (np.abs(zratios) >= 2).sum(axis=1)

        return np.c_[X, outlier_flag_counter]
# ________________________________________________________________________



# Load data from input folder
@st.cache(allow_output_mutation=True)
def load_data(uploaded_file,type):
    if type == 'csv':
        return pd.read_csv(uploaded_file)
    if type == 'xlsx':
        return pd.read_excel(uploaded_file)
    else:
        print('Invalid format')


# Score record by input form
@st.cache()
def score_record(input_values,model,predict_class = False, proba_treshold = 0.55):

    # return class based on cut off
    if predict_class:
        scores = (model.predict_proba(input_values)[:, 1] >= proba_treshold).astype(int)
    else:
        # return probabilities
        scores = model.predict_proba(input_values)[0]
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

@st.cache(allow_output_mutation=True)
def donut_chart(scores,label,min_rate ,max_rate):
    # Display prediction chart
    data = scores[0]
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.figure(frameon=False, figsize=(5, 5))
    plt.gcf().gca().add_artist(centre_circle)
    plt.pie(data, labels=label, textprops={'color': "white"},
            colors=['skyblue', 'darkred'], autopct='%1.1f%%', pctdistance=0.85, )

    risk_rate = optimal_rate(scores[0][1] ,min_rate ,max_rate )

    status = f" Optimal Rate \n {risk_rate}"

    text = plt.annotate(status,
                         xy=(0, 0),
                         fontsize=20,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color='black')
    return fig



best_features = ['Attr1',
                 'Attr24',
                 'Attr29',
                 'Attr34',
                 'Attr39',
                 'Attr40',
                 'Attr44',
                 'Attr46',
                 'Attr56',
                 'Attr60']

indicators=["net profit / total assets ",

            "gross profit (in 3 years) / total assets ",

            "logarithm of total assets ",

            "operating expenses / total liabilities",

            "profit on sales / sales",

            "(current assets - inventory - receivables) / short-term liabilities ",

            "(receivables * 365) / sales ",

            "(current assets - inventory) / short-term liabilities" ,

            "(sales - cost of products sold) / sales ",

            "sales / inventory "]

def optimal_rate(risk: float,min_rate= 6 , max_rate = 17):
    risk_limit = 0.45
    risk_cut = np.round(np.linspace(0.01, risk_limit, 10), 3)
    rate_cut = np.round(np.linspace(min_rate, max_rate, 10), 3)

    rate_dict = {risk: rate for risk, rate in zip(risk_cut, rate_cut)}

    if risk >= risk_limit:
        return "No credit"
    else:
        rate_min_risk = np.where(pd.Series(rate_dict).keys() < risk)[0][-1:]
        return pd.Series(rate_dict).iloc[rate_min_risk].values[0]


######################################################################


########################### Main function  ##########################

matrix_url = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fs-media-cache-ak0.pinimg.com%2Foriginals%2Fc5%2F9a%2Fd2%2Fc59ad2bd4ad2fbacd04017debc679ddb.gif&f=1&nofb=1"
cool_url = 'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwallpapercave.com%2Fwp%2Fwp6830287.gif&f=1&nofb=1'
cube_url = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwallpapercave.com%2Fwp%2Fwp2757874.gif&f=1&nofb=1"


def main():
    st.markdown(
        """
        <style>
        .reportview-container {
            background: url("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwallpapercave.com%2Fwp%2Fwp2757874.gif&f=1&nofb=1");transparency:0.8;
            background-position: 0px 0;
            background-size: 100%
           
        }
       .sidebar .sidebar-content {
            background: url("url_goes_here")
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    #  ![Alt Text](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fbeyondthearc.com%2Fwp-content%2Fmedia%2Fiwatsonanimated.gif&f=1&nofb=1)"


    #_________________________ Menu setting ________________________



    st.markdown('<center><h2>Ensemble Learning methods for corporate loans </h2></center>', unsafe_allow_html=True)

    # Menu
    menu = ["Getting Started","Single Form Prediction","Bulk Prediction"]
    activity = st.selectbox("Activity", menu)
    # _________________________________________________________________

    # pickle model
    model = pickle.load(open('model', 'rb'))


    if activity == "Getting Started":
        st.markdown('<h3>About: </h3>', unsafe_allow_html=True)

        st.write("This webapp takes as inpute a vector of financial ratios and evaluates the probability "
                 "of default for manufacturer companies.")
        st.write(
            "If the probability of default is below a certain alert value,"
            " it returns a weighted interest rate based on risk.")

        st.markdown('<h3>Benefits of Machine learning in selecting interest rates: </h3>', unsafe_allow_html=True)
        st.write(
            "The benefit of the machine learning approach is that you can incorporate the risk factor \n"
            "in determining the optimal interest rate.\n"
            "ML Models are very powerful and are much flexible in terms of retraining process than other techniques."
            )


        st.markdown("""

            <div style="text-align: center">
                <a href="link">
                    <img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fimages.squarespace-cdn.com%2Fcontent%2Fv1%2F541ba49ce4b0c158abe78bec%2F1569804288962-V6IS2TDHPO2IA8I4MZY0%2Fke17ZwdGBToddI8pDm48kDAv91l_EVpAtt3T82Wm1m9Zw-zPPgdn4jUwVcJE1ZvWEtT5uBSRWt4vQZAgTJucoTqqXjS3CfNDSuuf31e0tVEcs4OJ1MUiSygP0U4z2bUeJj0Nr1n48rGt1cKo_lK-mJuG45vQwBxdpDrCGUSSl5w%2Fcapabilities_feasibility.gif&f=1&nofb=1" align="center"></a>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<h3>How it works: </h3>', unsafe_allow_html=True)
        st.write(
            "Select in  **Activity** selectbox the way you want to upload your data, then provide the following information.")

        st.write(indicators)
        st.write("Once you have selected the way you want to use the app")
        st.write("1. Select the min and the max interest rates with the range bar.")
        st.write("2. Provide the data with the form or bulk csv.")
        st.write("3. Click on Predict button.")
        st.write("4. Get the optimal interest rates.")

    # ______________________ Prediction _____________________________
    if activity == "Single Form Prediction":

        # Form input text
        with st.form(key="Input"):

            min_rate = st.slider(label='Select the lower bound for interest rate', min_value=0, max_value=100, value=1,step=1)
            max_rate = st.slider(label='Select the upper bound for interest rate', min_value=min_rate, max_value=100, value=10,step=1)
            st.write("You need the following indicators:", indicators)
            text_input = st.text_area("Copy and paste data with a comma separator")
            st.write("Example: 0.0924,0.1948,4.4072,0.3879,0.1093,0.2659,38.766,0.7578,0.0643,6.0215 ")
            input_values = text_input.split(",")
            predict_button = st.form_submit_button("Predict")
            if int(max_rate) < int(min_rate):
                st.write("Your min is bigger than max, set again the range for interest rates.")
            try:
                input_values = pd.DataFrame(pd.Series(input_values).astype(float)).T
                input_values.columns = indicators
                st.write(input_values.T)
            except:
                pass
            if predict_button:
                with st.spinner('Wait for the prediction plot...'):
                    #___________________ Get prediction _______________
                    label = ['NoRisk', 'Risk']
                    scores = np.round( model.predict_proba(input_values.values) , 2 )
                    #Preview scores
                    st.text( scores )

                    # Display prediction chart
                    label = ['NoRisk','Risk']
                    #___________________________________________________

                    st.header("Prediction")

                    #_________________Donut chart settings ______________________________
                    fig = donut_chart( scores, label,min_rate ,max_rate)

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
        st.sidebar.write("You need the following indicators:")
        indicator_bars = pd.Series(data=indicators,name='Indicators')
        st.sidebar.table(indicator_bars)
        min_rate = st.slider(label='Select the lower bound for interest rate', min_value=0, max_value=100, value=1,step=1)
        max_rate = st.slider(label='Select the upper bound for interest rate', min_value=min_rate, max_value=100,
                             value=10, step=1)
        # Upload file
        # global df
        if uploaded_file is not None:
            try:
                df =  load_data(uploaded_file,'csv')[best_features]
            except Exception as e :
                print(e)
                df =  load_data(uploaded_file,'xlsx')[best_features]

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
                columns=['Probability negative','Probability positive']).sort_values(by = 'Probability negative'
                                          ,ascending=False)

                # Compute optimal rates scores
                ranked_risk["interest_rate"] = ranked_risk['Probability positive'].apply(lambda x:str(optimal_rate(x,min_rate,max_rate)))

                st.subheader('Ranked Precictions')
                st.write(ranked_risk)
                # Download results
                st.markdown(st_pandas_to_csv_download_link(ranked_risk, file_name="predictions.csv"), unsafe_allow_html=True)

                # Display highest
                st.header("Top 10 listed")
                top =ranked_risk.head(10)
                top = pd.concat([df.loc[top.index,:], top ],1)
                st.dataframe(top)
    # _________________________________________________________________

######################################################################



######################### RUN web app #################################
if __name__ == '__main__':
        main()
########################################################################


