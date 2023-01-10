# -*- coding: utf-8 -*-
"""
"""

'''Below libraries are to be installed and imported
 in the system for successful exection of this scripting file'''
#import yfinance as yf
from yahoofinancials import YahooFinancials
import quandl
import pandas as pd
#import numpy as np
#from datetime import datetime, date
#import datetime as dt
#import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
import plotly.graph_objects as plotg
from plotly.subplots import make_subplots
from plotly.offline import plot
#import math
#from sklearn.model_selection import train_test_split
import string
#from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.model_selection import cross_val_score
#from sklearn.metrics import mean_absolute_error,r2_score
import streamlit as st

quandl.ApiConfig.api_key = 'WByPs5dfCfxFQtHgD6uv'

class descriptive:
    def __init__(self):
        '''
        
        '''
        
    ###################################################################################################
    #______________________Previous function to download data for specified company___________________________##
    ###################################################################################################
    
    def get_prices(ticker, start_date, end_date, timeframe):
        # declaring info as global dictionary so that info can used everywhere post the function executions
        #global info
        info = {}
        
        # ''' 
        # - using YahooFinancials for fetching the stock data from yahoofinancial web basis start and end date
        # - ticker is the company
        # - start_date = starting date of the data to be fetched
        # - end_date = end date of the data to be fetched 
        
        prices = []
        #Download selected Ticker data
        try:
            yahoo_financials = YahooFinancials(ticker)
            print("here")
            historical_stock_prices1 = yahoo_financials.get_historical_price_data(start_date,
                                                                                 end_date, 
                                                                                 timeframe)
            #print("Hereeeee")
            prices = historical_stock_prices1[ticker]['prices']
            prices = pd.DataFrame(prices)
    
            #changing formatted_date object to datetime formate
            prices['formatted_date']=pd.to_datetime(prices['formatted_date'], format=('%Y-%m-%d'))
            prices = prices.drop('date', axis=1)
            #Capitalizing 1st letter of each column
            prices.columns = [string.capwords(i) for i in prices.columns ]
            prices = prices.rename(columns = {'Formatted_date':'Date','Adjclose':'Adjusted_close'}).set_index('Date')
            #prices.rename(columns = {'adjclose':'Adjusted_close'}).set_index('date')
            prices.insert(6, 'Returns', prices['Adjusted_close'].pct_change())
            print(prices)
            
        #Error message
        except:
            st.error('There is an error in downloading {} data for the period from {} to {} with {} frequency'.format(ticker, start_date, end_date, timeframe))
            st.stop()
        #Display the success message and the df as double check
        finally:
            print('{} data downloaded successfully.'.format(ticker))
            
            #inserting processed stock data of the company to info dictionary
            info[ticker] = prices
        
        '''
        Using SPY here in order to compare the stock price of the interested company with generic market.
        SPY - is a reflection of stock index of SPDR S&P 500
        '''
        try:
            spy_fin = YahooFinancials('SPY')
            historical_stock_prices2 = spy_fin.get_historical_price_data(start_date,
                                                                        end_date, 
                                                                        timeframe)
            
            spy = historical_stock_prices2['SPY']['prices']
            spy = pd.DataFrame(spy)
            spy['formatted_date']=pd.to_datetime(spy['formatted_date'], format=('%Y-%m-%d'))
            spy = spy.drop('date', axis=1)
            spy.columns = [string.capwords(i) for i in spy.columns ]
            spy = spy.rename(columns = {'Formatted_date':'Date','Adjclose':'Adjusted_close'}).set_index('Date')
            
            #spy = spy.rename(columns = {'formatted_date':'date'}).set_index('date')
            spy.insert(6, 'Returns', spy['Adjusted_close'].pct_change())
            
        #Error message
        except:
            st.error('There is an error in downloading SPY data for the period from {} to {} with {} frequency'.format(start_date,
                                                                                                             end_date,
                                                                                                             timeframe))
        finally:
            print('SPY data downloaded successfully.')
            #display(spy)
            
            #inserting spy stock indexes to the info dictionary
            info['spy'] = spy
        
        #risk free score using US treasury yield
        try:
            rf = quandl.get("USTREASURY/YIELD", start_date = start_date,
                                           end_date= end_date, collapse = timeframe)
    
            #Error message
        except:
            st.error('There is an error in downloading Risk free data for the period from {} to {} with frequency {}'.format(start_date,
                                                                                                             end_date,
                                                                                                             timeframe))
        finally:
            print('Risk free data downloaded successfully.')
            
            #inserting rf stock yield to the info dictionary
            info['risk_free'] = rf
        return info
    
    
    # data_dict=get_prices(ticker='AMZN', start_date='2018-08-13', end_date='2019-09-15', timeframe='daily')   
    
    ###################################################################################################
    #__________________________________Data Graph_____________________________________##
    ###################################################################################################
      
    #function for candlestick graph to plot    
    def candlestick_graph(ticker,data_dict, freq):
        #ticker: company whose candlestick is to be plotted
       try:
            df_ticker=pd.DataFrame(data_dict[ticker])
            
            if freq == 'weekly':
                df_ticker=df_ticker.resample('W').mean()
            elif freq == 'monthly':
                df_ticker=df_ticker.resample('M').mean()
            elif freq == 'yearly':
                df_ticker=df_ticker.resample('Y').mean()
            else:
                df_ticker=df_ticker = df_ticker
                
            #df_spy = data_dict['spy']
            fig=plotg.Figure(data=[plotg.Candlestick(x=df_ticker.index,
                        open=df_ticker['Open'],
                        high=df_ticker['High'],
                        low=df_ticker['Low'],
                        close=df_ticker['Close'])])
              
               
            fig.update_layout(
            xaxis =  {'showgrid': True},yaxis = {'showgrid': True} ,   
            title='Stock Price Movement',
            width=500,
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="#0E1117",
              font=dict(
                size=12,
                color="#A9A9A9" ), yaxis_title= ticker+' Stock')
            
            fig.update_xaxes(gridcolor='#404040', automargin=True)
            fig.update_yaxes(gridcolor='#404040', automargin=True)
       except:
           st.error('There was an error in loading of stock graphs. Try another stock.')
           st.stop()
       return (fig)
            #fig.write_image("images/fig1.png")
        
            
    #function for normal line graph to plot comparision between company and S&P 500
    
    def graph(ticker,var,data_dict):    
        '''
        -function for normal line graph to plot comparision between company and S&P 500
        -ticker : the company whose price is plotted
        -var : variable under consideration. closing price or adjusted closing price
        '''
        try:    
            df1 = pd.DataFrame(data_dict[ticker][var])
            df2 = pd.DataFrame(data_dict['spy'][var])
            df = df1.join(df2,how='inner',lsuffix = '_'+ticker, rsuffix = '_spy')
            fig1 = make_subplots(specs=[[{'secondary_y': True}]])
            #adding line plot for column 1
            fig1.add_trace(plotg.Scatter(x=df.index, y=df.iloc[:,0],name = df.columns[0]))
            #adding plot for column 2
            fig1.add_trace(plotg.Scatter(x=df.index, y=df.iloc[:,1],name = df.columns[1]),secondary_y=True)
            fig1.update_layout(
            xaxis =  {'showgrid': True},yaxis = {'showgrid': False},
            width=700,
            height=350,
            margin=dict(l=20, r=20, t=45, b=20),
            paper_bgcolor="#0E1117",
              font=dict(
                size=12,
                color="#A9A9A9" ),
            title="{}'s Adjusted closing price from {} to {}".format(ticker,  min(df1.index),max(df1.index)),
            yaxis_title ='adjusted closing price'
            ,
            legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="center",
            x=0.5)
            
            )
            
            fig1.update_xaxes(gridcolor='#404040')
            fig1.update_yaxes(gridcolor='#404040') 
        except:
            st.error('There was an error in loading graphs. Try another stock.')
        return (fig1)
    
    
      
    #_______________________________________END____________________________________________________
    ##################################################################################################
    
    
    ###################################################################################################
    #__________________________________Data Summary_____________________________________##
    ###################################################################################################
    #function for data summary
    def stats_summary(ticker,var,data_dict):
        '''
        getting the stock data from already fetched dictionary 'data_dict' to plot
        ticker : the company whose price is plotted
        var : variable under consideration. closing price or adjusted closing price
        '''
        try:
            df = data_dict[ticker]
            stats = pd.DataFrame(df[var].describe()) # fuction to get stats value
            stats=stats.T
            stats=stats.reset_index()
            stats=stats.rename(columns = {'index':'Variables'})
            stats['Range'] = stats['max']-stats['min']
            stats = stats.rename(columns = {'index':'Variables','mean':'Mean','std':'Standard Deviation','min':'Min Value','max':'Max Value',
                                            '25%':'25th Percentile','50%':'50th Percentile','75%':'75th Percentile'})
        except: 
            st.error('There is error in calculation of this stock. try another.')
            st.stop()
        return stats
    
           
    #_______________________________________END____________________________________________________
    ##################################################################################################
    
    
    #####################################################################################
    #___________________________________Simple moving average_____________________________________________
    #####################################################################################
    def calculate_sma(ticker,var,data_dict, order):
       
        ''' 
        - function to calculate the simple moving average of the stock data price
        - ticker is the company name at the stock exchange market,
        - var is the data variable used for computation. adjclose or return, open, close,..
        - order = is a list object containing all the MA(q) , by default order = [2,3,4,5,6,7,8,9,10]
        '''
        #Sort in order to avoid possible bugs
        try:
            order = sorted(order, reverse = True)
        except:
                print("data error")
        print('reached till here')
        print(ticker)
        print(var)
        #print(data_dict.keys())
        #Call the ticker
        df = data_dict[ticker]
        print(df)
        #Do a df with the different selected MA(q) for the adjclose
        try:
            sma_var = df[var]
            sma_var_series = pd.Series(sma_var)
            sma_df = pd.DataFrame(sma_var_series)
    
            for q in order:
                k = 1
                subsets = sma_var_series.rolling(q)
                moving_averages = subsets.mean()
                df_a = pd.DataFrame(moving_averages).rename(columns = {var:'{}_SMA_{}'.format(var,q)})
                sma_df.insert(k,'{}_SMA_{}'.format(var,q),df_a)
                k +=1
            
        #Error message
        except:
            print('There is an error in computing SMA for the ticker : {} using orders: {}'.format(ticker,order))
        
        finally:
            print("SMA for the ticker : {}'s adjclose computed successfully.".format(ticker))
        return sma_df
    
    
    #sma = calculate_sma('AMZN','Returns',data_dict,[3])
    
    
    def Sma_for_UI(self,data_dict,ticker,var,sma_order):
        '''
        -Function to be called by UI for plotting simple moving average with the origin variable
        -data_dict : main dictionary containing stock data
        -ticker : the company name 
        -sma_order : order(in list format) of simple moving average number to be computed. 
        '''
        try:
            df_sma = self.calculate_sma(ticker,var,data_dict,sma_order)
            fig2 = make_subplots(specs=[[{'secondary_y': True}]])
            #adding line plot for column 1
            fig2.add_trace(plotg.Scatter(x=df_sma.index, y=df_sma.iloc[:,0],name = df_sma.columns[0]))
            #adding plot for column 2
            fig2.add_trace(plotg.Scatter(x=df_sma.index, y=df_sma.iloc[:,1],name = df_sma.columns[1]),secondary_y=True)
            fig2.update_layout(
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
                'text' :"{}'s {} price from {} to {}".format(ticker, var, min(df_sma.index),max(df_sma.index)),
                'y':1.0,
                'x':0.5,
                'xanchor': "center",
                'yanchor': "top"},
            yaxis_title =var)
            
            
            fig2.update_xaxes(gridcolor='#404040')
            fig2.update_yaxes(gridcolor='#404040')

        except:
            st.error('There was an error in loading graphs for technical indicators. Try another stock.')
            st.stop()
        return (fig2)
        
    #__________________________________END______________________________________________
    #################################################################################### 
    
    
    #####################################################################################
    #___________________________________EMA_____________________________________________
    #####################################################################################
    def calculate_ema(ticker,var,data_dict, order = [10,9,8,7,6,5,4,3,2],smoothing = 2):
       
        ''' 
        - ticker is the company name at the stock exchange market,
        - var is the data variable used for computation. adjclose or return, open, close,..
        - order = is a list object containing all the MA(q) , by default order = [2,3,4,5,6,7,8,9,10]
        '''
        #Sort in order to avoid possible bugs
        order = sorted(order, reverse = False)
        
        #Call the ticker
        df = data_dict[ticker]
    
        #EMA for adjclose price calculation
    
        #Select the adjclose df
        var_ema = df[var]
        
        #adjclose_series = pd.Series(var_ema)
        df_ema_a = pd.DataFrame(var_ema)
        if var =='Returns':
            var_ema = df['Returns']
            var_ema = var_ema[1:]
    
        k=0
        try:
            for q in order:
                k += 1
                #i = -(k)
                ema = []
                
                #EMA calculation
                ema.append(sum(var_ema[:q]) / q)
                for price in var_ema[q:]:
                    ema.append((price * (smoothing / (1 + q))) + ema[-1] * (1 - (smoothing / (1 + q))))
                ema = pd.DataFrame(ema)
             
                if var =='Returns':
                    q=q+1
            
                #Set the first q raw NaN
                for x in range(-1,-q,-1):
                    ema.loc[x] = 'NaN'
                    
                ema = ema.sort_index()
                ema.index = ema.index + abs(q-1)
                                
                #Set te index and merge with main table
                ema = ema.set_index(df_ema_a.index)
                df_ema_a.insert(k,'{}_EMA({})'.format(var,q) ,ema)
            
        #Error message
        except:
            print('There is an error in computing EMA for the ticker : {} adjclose price using orders: {}'.format(ticker,order))
        
        finally:
            print("EMA for the ticker : {}'s adjclose price computed successfully.".format(ticker))
               
    
        return df_ema_a
        
    #ema = calculate_ema('AMZN','Returns',data_dict,order = [10,9])
     
    
    def Ema_for_UI(self,data_dict,ticker,var,ema_order):
        '''
        -Function to be called by UI for plotting exponential moving average with the origin variable
        -data_dict : main dictionary containing stock data
        -ticker : the company name 
        -sma_order : order(in list format) of simple moving average number to be computed. 
        '''
        
        try:
            df_ema = self.calculate_ema(ticker,var,data_dict,ema_order)
            #df_sma = data_dict['{}_{}_SMA'.format(ticker,price_type)]
            #df_sma = pd.DataFrame(df_sma)
            #df_sma = df_sma[[var,'sma_'+sma_order]]
            fig3 = make_subplots(specs=[[{'secondary_y': True}]])
            #adding line plot for column 1
            fig3.add_trace(plotg.Scatter(x=df_ema.index, y=df_ema.iloc[:,0],name = df_ema.columns[0]))
            #adding plot for column 2
            fig3.add_trace(plotg.Scatter(x=df_ema.index, y=df_ema.iloc[:,1],name = df_ema.columns[1]),secondary_y=True)
            fig3.update_layout(
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
                    'text' :"{}'s {} price Exponential MA from {} to {}".format(ticker, var, min(df_ema.index),max(df_ema.index)),
                    'y':1.0,
                    'x':0.5,
                    'xanchor': "center",
                    'yanchor': "top"},
                yaxis_title =var)
            fig3.update_xaxes(gridcolor='#404040')
            fig3.update_yaxes(gridcolor='#404040')
        except:
            st.error('There was an error in  loading graphs for technical indicators. Try another stock.')
            st.stop()
        return (fig3)

   
    # #_______________________________________END of Descriptive_project.py____________________________________________
    # ########################################################################################
    
   