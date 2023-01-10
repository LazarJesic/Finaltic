# -*- coding: utf-8 -*-
# """
# Project : Stock Symbol Group Project
# Author  :  Group 37 .
# Module : MIS41110
# """
#_________________Set Path_____________________________
#os.chdir(path_of_my_code)
#________________________________________________

#libararies required to import
import streamlit as st
import os

st.set_page_config(layout="wide")
#Set the path for project directory
original_dir = os.getcwd()
path = "C:/Users/Lazar/Desktop/Project_Final"
path_of_my_code = path
os.chdir(path_of_my_code)


from Predictive_project import predictive
from Descriptive_project import descriptive
#import socket
#from io import StringIO
#import BackEnd as back_end
import datetime as dt
#import pandas_datareader as pdr
import pandas as pd
#import yfinance as yf
import gc


today  = dt.date.today()
today = today.strftime("%Y-%m-%d")


###___________________________CLASS DEFINITION & ATTRIBUTES__ ___________________________
#########################################################################################

class UI:
    
    #Class UI  includes functions for webpage forms, descriptive, technicalindicator, predictive analysis
    
    
    def __init__(self, date_start, date_end, stocks_dropdown):
        #instantiates class attributes as per user input
        self.stocks_dropdown = stocks_dropdown
        self.data = pd.DataFrame()
        self.date_start = date_start.strftime('%Y-%m-%d')
        self.date_end = date_end.strftime('%Y-%m-%d')
        

    def get_data(self, stocks_dropdown, date_start, date_end, frequency):
        #loads data using module Descriptive_project.py and calls function get_prices()
        try:
            self.data = descriptive.get_prices(
                stocks_dropdown, date_start, date_end, frequency)
        except:
            st.error(
                'There was an error in data loading..check your network connection and try again.')
            st.stop()
        else:
            self.data = descriptive.get_prices(stocks_dropdown, date_start, date_end, frequency)
            

    def descriptive_section(self):
        #"""Displays Descriptive analysis and graphs"""
        try:
            Desc = st.expander( label='Expand/Collapse for Descriptive Analytics', expanded=True)
            #calling stats_summary function to get statistical analysis summary
            Desc.write(descriptive.stats_summary(
                ui.stocks_dropdown, ['Adjusted_close'], ui.data))
            Period = ['daily', 'weekly', 'yearly']
    
            Desc.write(
                '<style>div.row-widget.stRadio > div{flex-direction:row;} > .subheader{font-size:80px} </style>', unsafe_allow_html=True)
            period = Desc.radio(label="select the graph frequency", options=Period)
            
            #calling function  candlestick graph as per frequencies selected by the user
            
            if period == 'yearly':
                fig = descriptive.candlestick_graph(stocks_dropdown, ui.data, 'yearly')
                Desc.plotly_chart(fig, use_container_width=True)
            elif period == 'weekly':
                fig = descriptive.candlestick_graph(stocks_dropdown, ui.data, 'weekly')
                Desc.plotly_chart(fig, use_container_width=True)
            elif period == 'daily':
                fig = descriptive.candlestick_graph(stocks_dropdown, ui.data, 'daily')
                Desc.plotly_chart(fig, use_container_width=True)
    
            fig_s = descriptive.graph(stocks_dropdown, 'Adjusted_close', ui.data)
            Desc.plotly_chart(fig_s, use_container_width=True)
        except:
            st.error('There was an error in calculation of stock statistics/ graphs. Try another stock.')
            st.stop()

    def Technical_analysis(self):
        
        #This function references Descriptive_project.py and loads graphs for EMA, SMA from class descriptive
        
        try:
            Tech = st.expander(label='Expand/Collapse for Technical Analysis of stock', expanded=True)
            col1, col2 = Tech.columns(2)
            SMA_dropdown = col1.selectbox('Select period for SMA', options=[1, 2, 3, 4, 5, 6, 7, 8, 9])
            EMA_dropdown = col2.selectbox('Select period for EMA', options=[ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            fig_t_sma = descriptive.Sma_for_UI(descriptive, self.data, stocks_dropdown, 'Adjusted_close', [SMA_dropdown])
            fig_t_ema = descriptive.Ema_for_UI(descriptive, self.data, stocks_dropdown, 'Returns', [EMA_dropdown])
            #displaying graphs
            col1.plotly_chart(fig_t_sma, use_container_width=True)
            col2.plotly_chart(fig_t_ema, use_container_width=True)
        except:
            st.error('There was an error in calculation of stock technical indcators/graphs Try another stock.')     
            st.stop()

    def Predictive_analysis(self):
        
        #This function references module Predictive_project.py and loads Linear Regression, ARIMA, Gradient Boosting models 
        try:
            Pred = st.expander(label='Expand/Collapse for Predictive Analytics', expanded=True)
            timePeriods = [90, 270, 360, 450, 540, 630, 720]
    
            #This form, will take input dates for Predictive models 
            form_pred = Pred.form(key='input_predictions')
            with form_pred:
                col1, col2 = st.columns(2)
                date_forecast = col2.date_input("Forecast date", dt.date(2021, 12, 3))
                date_forecast = date_forecast.strftime('%Y-%m-%d')
                date_error_flag = date_error_handling(date_forecast,today )
                #takes forcasting date from user and checks for error handling 
                time_period_dropdown = col1.selectbox('Time Period for training the data', timePeriods)
                submit_button_pred = form_pred.form_submit_button(label='Get Predicted Price')
    
                if submit_button_pred: #is true/ is clicked
                    if(date_error_flag):
                        st.error('Predicting date should be greater than today')
                        st.stop()
                    else:
                        with st.spinner('Predicting for '+ui.stocks_dropdown):
                            ####### LINEAR REGRESSION MODEL #######
                            #calling function to get train & test data
                            model = predictive.train_test_data( predictive, stocks_dropdown, time_period_dropdown)
                            pred_df, fig_test, fig_act = predictive.linear_Nlinear_model( predictive,  model, stocks_dropdown, 'Adjusted_close', 'linear', date_forecast)
                            st.markdown("<h3 style ='text-align :center'>Linear Regression Model</h1>", unsafe_allow_html=True)
                            col3, col4 = st.columns(2)
                            #displaying the predicted output on the screen
                            col3.write("PREDICTED VALUE :   " + str(pred_df['predicted_value']), )
                            df = pd.DataFrame(pred_df['test_model_matrics'])
                            col4.write(df)
                            col31, col41 = st.columns(2)
                            #displaying graphs
                            col31.plotly_chart(fig_test, use_container_width=True)
                            col41.plotly_chart(fig_act, use_container_width=True)
            
                            ####### GRADIENT BOOSTING MODEL #########
                            #calling function to get test & train data
                            model2 = predictive.train_test_data(predictive, stocks_dropdown, time_period_dropdown)
                            pred_df_g, fig_test_g, fig_act_g = predictive.linear_Nlinear_model( predictive, model2, stocks_dropdown, 'Adjusted_close', 'non-linear', date_forecast)
                            st.markdown("<h3 style ='text-align :center'>Gradient Boosting Model</h1>", unsafe_allow_html=True)
                            col5, col6 = st.columns(2)
                            #displaying predicted output on screen
                            col5.write("PREDICTED VALUE :    " +str(pred_df_g['predicted_value']))
                            df = pd.DataFrame(pred_df_g['test_model_matrics'])
                            col6.write(df)
                            col51, col61 = st.columns(2)
                            #displaying graphs
                            col51.plotly_chart(fig_test_g, use_container_width=True)
                            col61.plotly_chart(fig_act_g, use_container_width=True)
                            
                            ######## ARIMA MODEL ########
                            #calling function to get train & test data
                            model3 = predictive.train_test_data(predictive, stocks_dropdown, time_period_dropdown)
                            pred_df_a, fig_test_a, fig_act_a = predictive.arima_model( predictive, model3, stocks_dropdown, 'Adjusted_close', date_forecast)
                            st.markdown("<h3 style = 'text-align : center;'>Arima Model</h1> ", unsafe_allow_html=True)
                            col7, col8 = st.columns(2)
                            df = pd.DataFrame({ 'Predicted Value': str(pred_df_a['forecate_value']), 'RMSE-S':[pred_df_a['test_rmse']],'MAPE':[pred_df_a['test_mape']]  ,'Accuracy':[pred_df_a['Accuracy']] }, )
                            df.reset_index(drop=True, inplace=True)
                            col8.write(df)
                            #displaying predicted value on screen
                            col7.write((" PREDICTED VALUE :\t "+ str(pred_df_a['forecate_value'])))
                            col71, col81 = st.columns(2)
                            #displaying graphs
                            col71.plotly_chart(fig_test_a, use_container_width=True)
                            col81.plotly_chart(fig_act_a, use_container_width=True)
        except: 
              st.error('There was an error in formulaiton of predictive model. Try another stock')
              st.stop()

# _____________________________FLOATING FUNCTION____________________
###########################################################################

#"""ERROR HANDLING"""
def date_error_handling(date_1, date_2):
    if date_1 == date_2 or date_1 < date_2:
        return 1
    else:
        return 0


#"""Function to get consolidated list of stocks so to avoid incorrect stock name as input"""

def load_stocks(file):
    #"""#reading company names from the text file containing list of available """
    stock_names=[]
    with open(file) as f:
        orgs = f.readlines()
    
    for i in range(len(orgs)):
        stock_names.append(orgs[i].rstrip('\n').split(','))
    stock = [item for sublist in stock_names for item in sublist]
    f.close()
    return stock



# ___________________________MAIN PROGRAM___________________________________
#############################################################################

# initialising the Class UI object using User input
# input_Stock is form for taking input values from the user

stock = load_stocks('stocks.txt')

st.markdown("<h1 style ='text-align :left'>STOCKER</h1>", unsafe_allow_html=True)
st.markdown("<style> .resize-triggers{ background-color: pink ;} </style>" ,unsafe_allow_html=True )

#input form for the user to give in values
with st.form(key='input_stock'):

    col1, col2, col3 = st.columns(3)
    date_start = col2.date_input("Start Date",dt.date(2019, 7, 6))

    date_end = col3.date_input("End Date",dt.date(2019, 7, 6))

    stocks_dropdown = col1.selectbox('Pick your Stock', stock)
    submit_button = st.form_submit_button(label='Get Stock Details')

if submit_button:
    date_error_flag = date_error_handling(date_end, date_start)
    if date_error_flag:
        st.error('start date and End date cannot be the same. Try again.')
        st.stop()

###Class UI Object instantiation
ui = UI(date_start, date_end, stocks_dropdown)

# code runs from top when it gets 2 dates same, it gives an error, hence this code is important
if(date_end == date_start):
    st.stop()

# load the data for input asked from user
with st.spinner('Fetching data for '+ui.stocks_dropdown):
    ui.get_data(ui.stocks_dropdown, ui.date_start, ui.date_end, 'daily')

# Descriptive Satistics
st.subheader('Descriptive Analytics')
ui.descriptive_section()
#Technical analysis
st.subheader('Technical Analytics')
ui.Technical_analysis()
#predictive analysis
st.subheader('Predictive Analytics')
ui.Predictive_analysis()


#clears garbage
gc.collect()
os.chdir(original_dir)
########################################## END OF MAIN #########################################################