# -*- coding: utf-8 -*-
"""

"""
#set working directory


'''Below libraries are to be installed and imported
 in the system for successful exection of this scripting file'''
#import yfinance as yf
import os
import quandl
import pandas as pd
import numpy as np
#from datetime import datetime, date
import datetime as dt
#import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
import plotly.graph_objects as plotg
from plotly.subplots import make_subplots
from plotly.offline import plot
import math
from statsmodels.tsa.arima_model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error,r2_score
from Descriptive_project import descriptive
#from Descriptive_project import Descriptive
quandl.ApiConfig.api_key = 'WByPs5dfCfxFQtHgD6uv'



class predictive():
    def __init__(self):
        
        '''
        
        '''
            
    ###################################################################################################
    #__________________________________Data Features_____________________________________##
    ###################################################################################################
    #pending-- need to design this basis the descriptive and modeling requirement
    def create_lag(df,var,lag_no):
        #temp = []
        for i in var:
            for j in lag_no:
                new_var = i+'_lag'+str(j)
                df[new_var] = df[i].shift(j)
            #temp = temp.append(df[var+'lg'+i])
        return df
        
    ###################################################################################################
    #__________________________________Data Preparation_____________________________________##
    ###################################################################################################
    
    def train_test_data(self,ticker,period):
        '''
        function to create train and test data on the basis of user input -- company, period of days
        period: no of days to be taken for training and testing purpose
        '''
        
        train_start_dt=dt.datetime.today() - dt.timedelta(days=period)
        train_end_dt= dt.datetime.today()
        
        train_start_dt = "{}-{}-{}".format(train_start_dt.year, train_start_dt.month , train_start_dt.day )
        train_end_dt = "{}-{}-{}".format(train_end_dt.year, train_end_dt.month , train_end_dt.day )
        print("checking")
        print(train_start_dt)
        print(train_end_dt)
        print(ticker)
        print(period)
        fetched_dict=descriptive.get_prices(ticker, start_date=train_start_dt, end_date=train_end_dt, timeframe="daily")
        
        df_to_model = fetched_dict[ticker]
        df_to_model = df_to_model.sort_index()
        #cheching if the dataset is valid and creating train and test data
        if len(df_to_model)>0:
            #creating some additional feature of the forecating variable to be used in models
            training_var = ['Adjusted_close','Returns']
            lag_feature = [1]
            df_to_model = self.create_lag(df_to_model,['Adjusted_close','Returns'],lag_feature)
            x_train, X_test, y_train, y_test = train_test_split(df_to_model, df_to_model.index, test_size=0.3, random_state=42,shuffle=False)
            #train = df[(df.index>=train_start_dt) & (df.index<=train_end_dt)]
            #test =  df[(df.index>=test_start_dt) & (df.index<=test_end_dt)]
        else:
            print("dataset is empty")         
        return [fetched_dict,df_to_model,x_train, X_test,training_var,lag_feature]    
    
   
    #_______________________________________END____________________________________________
    ########################################################################################
            
   
        
    ###################################################################################################
    #__________________________________Model Training_ARIMA_____________________________________##
    ###################################################################################################
    
    '''functions RMSE and MAPE are the test evaluation criteria basis rmse and mape score
    '''
    def RMSE(m,o):
        '''function to campute model's 'root mean square error' as a model performance matric '''
        return math.sqrt(((m-o)**2).mean())
        
    def MAPE(r,o):
        '''function to campute model's 'mean absolute percentage error'' as a model performance matric '''
        return abs((r-o)/r).mean()
        
    
    
    
    def test_actual_forcast(fcst,bandwidth,title='Actuals and Forecast'):
        '''
        function to plot the graph between closing price of  test data and forecats on test data'''
        fig4_1 = make_subplots(specs=[[{'secondary_y': True}]])
        #adding line plot for column 1
        fig4_1.add_trace(plotg.Scatter(x=fcst.index, y=fcst.iloc[:,0],name = fcst.columns[0]))
        #adding plot for column 2
        fig4_1.add_trace(plotg.Scatter(x=fcst.index, y=fcst.iloc[:,1],name = fcst.columns[1]))                                                                       
        fig4_1.update_layout(
        xaxis =  {'showgrid': True},yaxis = {'showgrid': True},
        yaxis_title ='Adjusted closing price',
      
        width = 500,
        
        height =300,
        
        margin=dict(l=10, r=10, t=10, b=10, pad =0),
        legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1,
        xanchor="center",
        x=0.5),
        paper_bgcolor="#0E1117",
          font=dict(
            size=10,
            color="#A9A9A9" ),
            title={
            'text' :title,
            'y':1.0,
            'x':0.5,
            'xanchor': "center",
            'yanchor': "top"},
        )
        
        
        fig4_1.update_xaxes(gridcolor='#404040')
        fig4_1.update_yaxes(gridcolor='#404040')
        
        return fig4_1
    
    
    def actual_forcast_plot(fcst,ticker,data_dict,bandwidth,title='Forecast and Actuals'):
        '''
        function to plot the graph between closing price of  complete training data and future forecat values'''
        fig4_2 = make_subplots(specs=[[{'secondary_y': True}]])
        #adding line plot for column 1
        df_ticker=data_dict[ticker][['Adjusted_close']]
        
        fig4_2.add_trace(plotg.Scatter(x=df_ticker.index, y=df_ticker.iloc[:,0],name = df_ticker.columns[0]))
        #adding plot for column 2

        if bandwidth=='yes':                                                         
            fig4_2.add_trace(plotg.Scatter(
                x=np.concatenate([fcst.index,fcst.index[::-1]]), 
                y=np.concatenate([fcst['upper_series'],fcst['lower_series'][::-1]]),
                fill='toself',mode = 'none'))
       
        fig4_2.add_trace(plotg.Scatter(x=fcst.index, y=fcst.iloc[:,0],name = fcst.columns[0],mode = 'lines',line=dict(color='firebrick',dash='dot',width= 2)))
    
                                                                     
        fig4_2.update_layout(
        xaxis =  {'showgrid': True},yaxis = {'showgrid': True},
        yaxis_title ='Adjusted closing price',

        width = 500,
        
        height =300,
        
        margin=dict(l=10, r=10, t=10, b=10, pad =0),
        legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1,
        xanchor="center",
        x=0.5),
        paper_bgcolor="#0E1117",
          font=dict(
            size=10,
            color="#A9A9A9" ),
            title={
            'text' :title,
            'y':1.0,
            'x':0.5,
            'xanchor': "center",
            'yanchor': "top"}
        )
        
        
        fig4_2.update_xaxes(gridcolor='#404040')
        fig4_2.update_yaxes(gridcolor='#404040')
      
        return fig4_2
        
        
    def arima_model(self,model_data,ticker,frcst_var,date_to_forecast):
        
        out_dict = {}
        model_dict= model_data[0]
        train= pd.DataFrame(model_data[2][frcst_var])
        test= pd.DataFrame(model_data[3][frcst_var])
    
        
        #______________________________________________________________
        
        # Build ARIMA Model
        
        '''For now fixing the p,d,q values for the model automation purpose'''

        model = ARIMA(train[frcst_var], order=(1,1,1))  
        
        fitted = model.fit(disp=-1)
        test_model_summary= fitted.summary() 
        out_dict['test_model_summary'] = test_model_summary
        
        # Forecast
        test_forecast, test_se, test_conf = fitted.forecast(len(test), alpha=0.05)  # 95% conf
        
        # Make as pandas series
        test['forecast'] = test_forecast
    
        out_dict['test_fcst'] = test
            
        rmse_score = self.RMSE(test['forecast'],test[frcst_var]) 
        mape_score = self.MAPE(test['forecast'],test[frcst_var]) 
        out_dict['test_rmse'] = round(rmse_score,2)
        out_dict['test_mape'] = round(mape_score,2)
        out_dict['Accuracy'] = (1-round(mape_score,2))*100
        
        '''calling the graph to plot actual test data and '''
        plt_test = self.test_actual_forcast(fcst = test,bandwidth = 'no',title = 'Actual Vs Forecasted adjusted closing price (Test data)')
        
            
        '''________________________Final model to forecast future dates given by date_________________'''
        # final Forecast as per user's requirement
        #creating the dataframe matrics for the future forecasting by using the user's forecast date
       
        fct_date = date_to_forecast
        out_dict['forecast_date'] = fct_date
        print(fct_date)
        fct_date=pd.to_datetime(fct_date, format = '%Y-%m-%d')
        '''handle error if forecast date is less then today'''
        if max(model_dict[ticker].index)<fct_date:
            day_to_frcst=(fct_date-max(model_dict[ticker].index)).days
        else:
            print('forecast date is less than today')       
        
        fcst_df = pd.DataFrame( data = 
            [max(model_dict[ticker].index)+dt.timedelta(days=d) for d in range(1,day_to_frcst+1)],
            columns = ['date'])
        
        fcst_df=fcst_df.set_index('date')
        
        
        '''traning on arima model'''
        model_final = ARIMA(model_dict[ticker][frcst_var], order=(1, 1, 1))  
        fitted_final = model_final.fit(disp=-1)
        '''predicting using arima model'''
        fc1, se1, conf1 = fitted_final.forecast(len(fcst_df), alpha=0.05)  # 95% conf
        

        fcst_df['forecast'] = fc1
        fcst_df['upper_series'] = conf1[:,0]
        fcst_df['lower_series'] = conf1[:,1]
        #fcst_df is the complete forecast for all the data periods till date to forecast
        out_dict['forecated_fset'] = fcst_df
        #forecast val
        #print(fcst_df)
        out_dict['forecate_value'] = round(fcst_df['forecast'][len(fcst_df)-1],2)
        
        '''calling the graph to plot actual price and future forecasted price'''
        fcst_plot = self.actual_forcast_plot(fcst_df,ticker,model_dict,'yes',title = 'Actual and Forecasted adjusted closing price')
        #fcst,ticker,data_dict,bandwidth,title
        #out_dict['forecate_plot'] = fcst_plot
        out_dict['model_dict'] = model_dict
        
        #print(fitted_final.summary())
        out_dict['fcst_model_summary'] = fitted_final.summary()
        
        return [out_dict,(plt_test),(fcst_plot)]

    #function call to plot test validation plot. plot between actual test values and model forecasted values.
    
    ################################### END Of ARIMA #############################
    
    ######################Functions for model fitting and calculation of model performance####################################################
        
    def return_metric(y_test,y_pred):
        '''
        -calculating mean average error
        -Trained model R square value
        -Mean average percentage error for accuracy calculation
        '''
        mae = mean_absolute_error(y_test,y_pred)
        r2 = r2_score(y_test,y_pred) 
        y_test, y_pred = np.array(y_test), np.array(y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test))
        print("mae:%s,r2:%s,mape:%s"%(mae,r2,mape))
        return mae,r2,mape
    
    def get_model_performance(self,model,X_train,X_test,y_train,y_test):
        '''
        -traing the model
        -testing the model performance using 10 fold cross validation results
        -predicting the values on test dataset
        -calculating the performance matrics to validate model performance on test dataset
        '''
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("cross validation score mean absolute error")
        print(cross_val_score(model,X_train, y_train, cv=10,scoring='neg_mean_absolute_error'))
        print("cross validation score r2")
        print(cross_val_score(model,X_train, y_train, cv=10,scoring='r2'))
        print("score on Test data:")
      
        perf_matrics = pd.DataFrame()
        #calling function to calculate model performance matrices 
        matric1,matric1,matric3 = self.return_metric(y_test,predictions)
        test_pred_df = pd.DataFrame(y_test)
        test_pred_df['predicted'] = predictions
        perf_matrics['MAE'] = [round(matric1,2)]
        perf_matrics['R-square'] = [round(matric1,2)]
        perf_matrics['MAPE'] = [round(matric3,2)]
        perf_matrics['Accuracy'] = [(1-round(matric3,2))*100]
        print(perf_matrics)
        return model,test_pred_df,perf_matrics
    
    ######################################################################################
    ''' model buil for linear regression and gradient boosting tree based algorithm '''
    ######################################################################################
    
    
    def linear_Nlinear_model(self,model_data,ticker,fcst_var,model_type ,date_to_forecast ):
        '''
        creating a fuction to execute Linear Regression model as well as 
        Non Linear tree based 'Gradient boosting' algorithm' for future prices prediction
        
        -for Regression model ,model_type = 'non-linear'
        -for non linear tree based gradient boosting model, model_type = 'non-linear'
    
        '''
    
        out_dic={}
        #preparing the data for gradient boosting model -non linear model
        total_df = model_data[1]
        x_train = model_data[2].dropna().sort_index()
        x_test = model_data[3].dropna().sort_index()
    
        x_train_df=x_train[[x  for x in x_train.columns if('lag1' in x)]]
        y_train_df = x_train[fcst_var]
        x_test_df=x_test[[x  for x in x_train.columns if('lag1' in x)]]
        y_test_df = x_test[fcst_var]
        
        '''training and testing model on the splitted train and test data respectively'''
        if model_type == 'Linear':
            model = LinearRegression()
        else:
             model = GradientBoostingRegressor()
        model_regression,test_predicted,test_pred_matrics = self.get_model_performance(self,model,x_train_df,x_test_df,y_train_df,y_test_df)
       
        out_dic['test_predicted'] = test_predicted
        out_dic['test_model_matrics'] = test_pred_matrics
        '''Plotting test vs predicted.'''
        fig5 = make_subplots()
        #adding line plot for column 1
        fig5.add_trace(plotg.Scatter(x=test_predicted.index, y=test_predicted.iloc[:,0],name = test_predicted.columns[0]))
        #adding plot for column 2
        fig5.add_trace(plotg.Scatter(x=test_predicted.index, y=test_predicted.iloc[:,1],name = test_predicted.columns[1]))
        fig5.update_layout(
        xaxis =  {'showgrid': False},yaxis = {'showgrid': False},
        title="{}'s Predicted versus Test closing price from {} to {}".format(ticker,  min(test_predicted.index),max(test_predicted.index)),
        yaxis_title ='Adjusted closing price')
        #(fig5)
        
    
        '''Precting for future dates'''
        #creating the dataframe matrics for the future forecasting by using the user's forecast date
        fct_date = date_to_forecast
        out_dic['forecast_date'] =date_to_forecast
    
        fct_date=pd.to_datetime(fct_date, format = '%Y-%m-%d')
        day_to_frcst=(fct_date-max(total_df.index)).days
            
        fcst_df = pd.DataFrame( data = 
                [max(total_df.index)+dt.timedelta(days=d) for d in range(1,day_to_frcst+1)],
                columns = ['date'])
        fcst_df=fcst_df.set_index('date')
        
        '''now using the complete available training data to retrain the model to predict future unknown dates'''
        final_train_df = total_df.dropna()
        train_var = [x  for x in final_train_df.columns if('lag' in x)]
        final_x_train = final_train_df[train_var]
        final_y_train = final_train_df[fcst_var]
        print(fcst_df)
        final_model = LinearRegression()
        final_model.fit(final_x_train, final_y_train)
        #predictions = final_model.predict(X_test)
        #appending last record from available data before forecasting to get lag(X values) element of first forecasting index
        x=pd.DataFrame(final_train_df.iloc[-1,]).T
        pred_df=x.append(fcst_df,ignore_index=False)
        
        '''Prediction For the model with lag 1 feature.
        Step forecasting approach have been used to maintain the accuracy and consistency of the forecast'''
        
        for i in range(0,len(pred_df)-1,1):
            for var in train_var:
                base = var.split(sep = '_lag')[0]
                pred_df[var][i+1] = pred_df[base][i]
            pred = final_model.predict(pred_df[train_var][i+1:i+2])
            pred_df[fcst_var][i+1] = pred
            if pred_df[fcst_var][i] !=0:
                pred_df['Returns'][i+1] = (pred-pred_df[fcst_var][i])/(pred_df[fcst_var][i])
            else:
                pred_df['Returns'][i+1] = np.NaN
            
        pred_df = pred_df.iloc[1:len(pred_df)-1]
        predicted_val =  pred_df[fcst_var][len(pred_df)-1]  
        
        out_dic['predicted_value'] = round(predicted_val,2)
        out_dic['full_prediction_df'] = pred_df
        total_df[[fcst_var]]
        pred_df = pred_df.rename(columns={'Adjusted_close':'Predited_close'})
        train_pred = pd.DataFrame(total_df[[fcst_var]].append(pred_df[['Predited_close']]))
        
        fig6 = make_subplots()
        #adding line plot for column 1
        fig6.add_trace(plotg.Scatter(x=train_pred.index, y=train_pred.iloc[:,0],name = train_pred.columns[0]))
        #adding plot for column 2
        fig6.add_trace(plotg.Scatter(x=train_pred.index, y=train_pred.iloc[:,1],name = train_pred.columns[1],line=dict(color='red',dash="dot",width= 2)))
        fig5.update_layout(
            xaxis =  {'showgrid': True},yaxis = {'showgrid': True},
            width = 500,
            height =300,
              margin=dict(l=10, r=10, t=10, b=10, pad =0),
            legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="center",
            x=0.5),
             paper_bgcolor="#0E1117",
              font=dict(
                size=10,
                color="#A9A9A9" ),
               title={
                'text' :"{}'s Predicted versus Test closing price from {} to {}".format(ticker,  min(test_predicted.index),max(test_predicted.index)),
                'y':1.0,
                'x':0.5,
                'xanchor': "center",
                'yanchor': "top"},
            yaxis_title ='Adjusted closing price')
    
    
        fig6.update_layout(
        xaxis =  {'showgrid': True},yaxis = {'showgrid': True},
        width = 500,
        height =300,
          margin=dict(l=10, r=10, t=10, b=10, pad =0),
        legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1,
        xanchor="center",
        x=0.5),
         paper_bgcolor="#0E1117",
          font=dict(
            size=10,
            color="#A9A9A9" ),
           title={
            'text' :"{}'s Predicted versus Test closing price from {} to {}".format(ticker,  min(test_predicted.index),max(test_predicted.index)),
            'y':1.0,
            'x':0.5,
            'xanchor': "center",
            'yanchor': "top"},
        yaxis_title ='Adjusted closing price')
        
        fig5.update_xaxes(gridcolor='#404040')
        fig5.update_yaxes(gridcolor='#404040')
        
        fig6.update_xaxes(gridcolor='#404040')
        fig6.update_yaxes(gridcolor='#404040')
        return [out_dic,(fig5),(fig6)]
    
############################################# END OF Predictive_Project.py ###################################
        
