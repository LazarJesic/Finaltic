from yahoofinancials import YahooFinancials
import quandl
import string
import datetime
import numpy as np
import pandas as pd
import numpy as np
from tqdm import tqdm
#All necessary plotly libraries
import plotly as py
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from tqdm import tqdm
import statsmodels.graphics.tsaplots as sm
#Scikit-Learn for Modeling
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error,mean_squared_log_error
#Statistics
import statsmodels.api as sm
from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing
from pmdarima import auto_arima
from datetime import timedelta



def spy_prices(start_date,end_date,timeframe):
    '''Function to always call back the '''
    yahoo_financials = YahooFinancials('SPY')
    historical_stock_prices1 = yahoo_financials.get_historical_price_data(start_date,
                                                                         end_date, 
                                                                         timeframe)            
    prices = historical_stock_prices1['SPY']['prices']
    prices = pd.DataFrame(prices)
    #changing formatted_date object to datetime formate
    prices['formatted_date']=pd.to_datetime(prices['formatted_date'], format=('%Y-%m-%d'))
    prices = prices.drop('date', axis=1)
    #Capitalizing 1st letter of each column
    prices.columns = [string.capwords(i) for i in prices.columns ]
    prices = prices.rename(columns = {'Formatted_date':'Date','Adjclose':'Adjusted_close'}).set_index('Date')
    #prices.rename(columns = {'adjclose':'Adjusted_close'}).set_index('date')
    prices.insert(6, 'Returns', prices['Adjusted_close'].pct_change())
    return prices

def spy_stats(var,start_date,end_date,timeframe):
    '''
    getting the stock data for SPY
    var : variable under consideration. closing price or adjusted closing price
    '''
    spy = spy_prices(start_date,end_date,timeframe)
    stats = pd.DataFrame(spy[var].describe()) # fuction to get stats value
    stats=stats.T
    stats=stats.reset_index()
    stats=stats.rename(columns = {'index':'Variables'})
    stats['Range'] = stats['max']-stats['min']
    stats = stats.rename(columns = {'index':'Variables','mean':'Mean','std':'St. Deviation','min':'Min Value','max':'Max Value',
                                    '25%':'25th Percentile','50%':'50th Percentile','75%':'75th Percentile'})
    stats = stats.drop('Variables',axis=1)
    return stats

class Stock():
    '''
    Stock object, made in order to dornload, transform and extract statistics from the selected ticker
    '''
    def __init__(self, ticker):
        self.name = ticker
        self.weight = 0
    
    def get_prices(self,start_date,end_date,timeframe):
        prices = []
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        #Download selected Ticker data
        try:
            yahoo_financials = YahooFinancials(self.name)
            historical_stock_prices1 = yahoo_financials.get_historical_price_data(start_date,
                                                                                 end_date, 
                                                                                 timeframe)            
            prices = historical_stock_prices1[self.name]['prices']
            prices = pd.DataFrame(prices)

            #changing formatted_date object to datetime formate
            prices['formatted_date']=pd.to_datetime(prices['formatted_date'], format=('%Y-%m-%d'))
            prices = prices.drop('date', axis=1)
            #Capitalizing 1st letter of each column
            prices.columns = [string.capwords(i) for i in prices.columns ]
            prices = prices.rename(columns = {'Formatted_date':'Date','Adjclose':'Adjusted_close'}).set_index('Date')
            #prices.rename(columns = {'adjclose':'Adjusted_close'}).set_index('date')
            prices.insert(6, 'Returns', prices['Adjusted_close'].pct_change())
            self.prices = prices
        #Error message
        except:
            print(f'There is an error in downloading {self.name} data for the period from {start_date} to {end_date} with {timeframe} frequency')
    
    def get_statistics(self,var):
        '''
        getting the stock data from already fetched dictionary 'data_dict' to plot
        ticker : the company whose price is plotted
        var : variable under consideration. closing price or adjusted closing price
        '''
        try:
            stats = pd.DataFrame(self.prices[var].describe()) # fuction to get stats value
            stats=stats.T
            stats=stats.reset_index()
            stats=stats.rename(columns = {'index':'Variables'})
            stats['Range'] = stats['max']-stats['min']
            stats = stats.rename(columns = {'index':'Variables','mean':'Mean','std':'St. Deviation','min':'Min Value','max':'Max Value',
                                    '25%':'25th Percentile','50%':'50th Percentile','75%':'75th Percentile'})
            stats = stats.drop('Variables',axis=1)
            self.stats = stats
            return stats
        except: 
            print('There is error in calculation of this stock. try another.')
            
            
    def get_SPYgraph(self):
        
        spy = spy_prices(self.start_date,self.end_date,self.timeframe)
        df = pd.concat([spy['Returns'].rename('SPY'),self.prices['Returns'].rename(self.name)],axis=1, join='inner')
        fig = px.line(df, x=df.index, y=df.columns)
        fig.update_layout(
        xaxis =  {'showgrid': False},yaxis = {'showgrid': False} ,   
        title=f'{self.name} vs SPY Returns comparison',
        width=650,
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="#fff",
          font=dict(
            size=12,
            color="rgb(134,141,151)" ), yaxis_title= self.name +' Returns ')
        
        fig.update_xaxes(gridcolor='rgb(134,141,151)', automargin=True)
        fig.update_yaxes(gridcolor='rgb(134,141,151)', automargin=True)
        fig.update_layout({
            'plot_bgcolor': '#fff',
            'paper_bgcolor': '#fff',
        }
        ,title={
        'y':1,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
        },
        xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
        ))
        fig.data[0].line.color = "#a83b3b"
        fig.data[1].line.color ="#886ab5"
        return fig

    def get_candlestick(self):
        df_ticker=self.prices
        df_ticker=df_ticker.resample('W').mean()
        fig = go.Figure(data=[go.Candlestick(x=df_ticker.index,
                    open=df_ticker['Open'],
                    high=df_ticker['High'],
                    low=df_ticker['Low'],
                    close=df_ticker['Close'])])
          
           
        fig.update_layout(
        xaxis =  {'showgrid': False},yaxis = {'showgrid': False} ,   
        title='Weekly Average Stock Price Movement',
        width=600,
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="#E5E7EB",
          font=dict(
            size=12,
            color="rgb(134,141,151)" ), yaxis_title= self.name +' Stock')
        
        fig.update_xaxes(gridcolor='rgb(134,141,151)', automargin=True)
        fig.update_yaxes(gridcolor='rgb(134,141,151)', automargin=True)
        fig.update_layout({
            'plot_bgcolor': '#fff',
            'paper_bgcolor': '#fff',
        }
        ,title={
        'y':1,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
        },
        xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
        ))

        return fig

    def get_CAPM(self):
         #Function to return the linear regression between market returns and assets returns
       
        spy = spy_prices(self.start_date,self.end_date,self.timeframe)
        LR = stats.linregress(self.prices['Returns'].iloc[1:],spy['Returns'].iloc[1:])
        return LR

    def get_SPYcorr(self):
        
        spy = spy_prices(self.start_date,self.end_date,self.timeframe)
        fig = px.scatter(x=self.prices['Returns'], y=spy['Returns'], title='',  trendline="ols")
        fig.update_layout(
        xaxis =  {'showgrid': False},yaxis = {'showgrid': False} ,   
        title=f'{self.name} and SPY Correlations',
        width=650,
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="#fff",
          font=dict(
            size=12,
            color="rgb(134,141,151)" ), yaxis_title= self.name +' Returns ')
        
        fig.update_xaxes(gridcolor='rgb(134,141,151)', automargin=True)
        fig.update_yaxes(gridcolor='rgb(134,141,151)', automargin=True)
        fig.update_layout({
            'plot_bgcolor': '#fff',
            'paper_bgcolor': '#fff',
        }
        ,title={
        'y':1,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
        },
        xaxis_title="SPY Returns",
        )
        results = px.get_trendline_results(fig)
        fig.data[1].line.color ="#495057"
        fig.update_traces(marker=dict(
        color='#886ab5'))
        return fig,results

    def  test_stationarity(self):
        #Determing rolling statistics
        timeseries = self.prices['Close']
        rolmean = timeseries.rolling(12).mean()
        rolstd = timeseries.rolling(12).std()
        df = pd.concat([timeseries.rename('timeseries'),rolmean.rename('rolmean'),rolstd.rename('rolstd')],axis=1, join='inner').dropna()
        #Plot rolling statistics:
        fig = px.line(df, x=df.index, y=df.columns)
        fig.update_layout(
        xaxis =  {'showgrid': False},yaxis = {'showgrid': False} ,   
        title=f'{self.name} Closed price, Rolling mean and Rolling standard deviation',
        width=650,
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="#fff",
          font=dict(
            size=12,
            color="rgb(134,141,151)" ), yaxis_title= 'Value')
        
        fig.update_xaxes(gridcolor='rgb(134,141,151)', automargin=True)
        fig.update_yaxes(gridcolor='rgb(134,141,151)', automargin=True)
        fig.update_layout({
            'plot_bgcolor': '#fff',
            'paper_bgcolor': '#fff',
        }
        ,title={
        'y':1,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
        },
        xaxis_title="Date",
        )
        results = px.get_trendline_results(fig)
        fig.data[1].line.color ="#495057"
        fig.data[0].line.color ="#886ab5"
        fig.data[2].line.color ="#a83b3b"
        fig.update_traces(marker=dict(
        color='#886ab5'))
        return fig

    def ARIMAAll(self):
        df = self.prices
        df['Close_diff'] = df['Close'].diff()
        df['Close_diff_log'] = np.log1p(df['Close'])-np.log1p(df.shift()['Close'])
        model_train=df.iloc[:int(df.shape[0]*0.80)]
        valid=df.iloc[int(df.shape[0]*0.80):]
        y_pred=valid.copy()
        model_scores_r2=[]
        model_scores_mse=[]
        model_scores_rmse=[]
        model_scores_mae=[]
        model_scores_rmsle=[]
        model_arima= auto_arima(y=model_train["Close"].dropna(), trace=True, error_action='ignore', start_p=1,start_q=1,max_p=10,max_q=10,
                    suppress_warnings=True,stepwise=False,seasonal=False)
        arima = model_arima.fit(model_train["Close"].dropna())
        prediction_arima=model_arima.predict(len(valid))
        prediction_arima.index = pd.to_datetime(y_pred.index)
        y_pred["ARIMA Model Prediction"]=prediction_arima
        r2_arima= r2_score(y_pred["Close"],y_pred["ARIMA Model Prediction"])
        mse_arima= mean_squared_error(y_pred["Close"].dropna(),y_pred["ARIMA Model Prediction"].dropna())
        rmse_arima=np.sqrt(mean_squared_error(y_pred["Close"],y_pred["ARIMA Model Prediction"]))
        mae_arima=mean_absolute_error(y_pred["Close"],y_pred["ARIMA Model Prediction"])
        rmsle_arima = np.sqrt(mean_squared_log_error(y_pred["Close"],y_pred["ARIMA Model Prediction"]))
        model_scores_r2.append(r2_arima)
        model_scores_mse.append(mse_arima)
        model_scores_rmse.append(rmse_arima)
        model_scores_mae.append(mae_arima)
        model_scores_rmsle.append(rmsle_arima)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=model_train.index, y=model_train["Close"], mode='lines',name="Train Data for Stock Prices"))
        fig.add_trace(go.Scatter(x=valid.index, y=valid["Close"], mode='lines',name="Validation Data for Stock Prices",))
        fig.add_trace(go.Scatter(x=valid.index, y=y_pred["ARIMA Model Prediction"], mode='lines',name="Prediction for Stock Prices",))
        fig.update_layout(title="ARIMA",xaxis_title="Date",yaxis_title="Close",legend=dict(x=0,y=1,traceorder="normal"),font=dict(size=12))

        fig.update_layout(
        xaxis =  {'showgrid': False},yaxis = {'showgrid': False} ,   
        title=f'{self.name} ARIMA prediction',
        width=650,
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="#fff",
          font=dict(
            size=12,
            color="rgb(134,141,151)" ), yaxis_title= 'Value')
        
        fig.update_xaxes(gridcolor='rgb(134,141,151)', automargin=True)
        fig.update_yaxes(gridcolor='rgb(134,141,151)', automargin=True)
        fig.update_layout({
            'plot_bgcolor': '#fff',
            'paper_bgcolor': '#fff',
        }
        ,title={
        'y':1,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
        },
        xaxis_title="Date",
        )
        fig.data[1].line.color ="#495057"
        fig.data[0].line.color ="#886ab5"
        fig.data[2].line.color ="#a83b3b"
        fig.update_traces(marker=dict(
        color='#886ab5'))
        return arima, fig, r2_arima, mse_arima, mae_arima

class Portfolio():
    '''
    A class portfolio with arguments tickers
    '''
    def __init__(self,tickers):
        self.tickers = tickers
    
    def makeStocks(self,start,end,timeframe):
        #function made in order to create different stocks value
        try:
            stocks = []
            for ticker in self.tickers:
                ticker = Stock(ticker)
                print(ticker)
                ticker.get_prices(start,end,timeframe)
                ticker.get_statistics('Returns')  
                stocks.append(ticker)
                self.stocks = stocks
        except:
            print('error')
            
    def calculateReturns(self,weights):
       # if sum(weights) != 1 :
       #     print(f"The portfolio percentage doesn't sum to 100%.\nYou are {'over' if sum(weights)>1 else 'under'} by {round((1-sum(weights))*100,2)}%")
       # #Function to calculate the weighted portfolio returns 
        try:
            dfReturns = pd.DataFrame(self.stocks[0].prices['Returns'])
            for stock in range(1, len(self.stocks)):
                returns = pd.DataFrame(self.stocks[stock].prices['Returns'])
                type(returns)
                dfReturns = pd.concat([dfReturns,returns],axis=1, join='inner')
            dfReturns.columns = [stock.name for stock in self.stocks]
            dfWReturns = (weights * dfReturns)
            port_ret = dfWReturns.sum(axis=1)
            self.returns = port_ret 
            self.dfWReturns = dfWReturns
            self.dfReturns = dfReturns
            #Get anualized return
            self.mus = (1+self.dfReturns.mean()) ** 252 - 1
            self.cov = self.dfReturns.cov()*252
        except:
            print('There was an error')
            
    def portfolioOptimization(self,n_assets,samples):
        n_assets = n_assets
        samples = samples
        #empty list to store mean-variance pairs
        pairs = []
        weights_list=[]
        tickers_list=[]
        np.random.seed(20) #Reproducibility
        for sample in tqdm(range(samples)):
            next_i = False
            while True:
                #- Choose assets randomly without replacement
                assets = np.random.choice(list(self.dfReturns.columns), n_assets, replace=False)
                #- Choose weights randomly ensuring they sum to one
                weights = np.random.rand(n_assets)
                weights = weights/sum(weights)
        
                #-- Loop over asset pairs and compute portfolio return and variance
                portfolio_E_Variance = 0
                portfolio_E_Return = 0
                for i in range(len(assets)):
                    portfolio_E_Return += weights[i] * self.mus.loc[assets[i]]
                    for j in range(len(assets)):
                        portfolio_E_Variance += weights[i] * weights[j] * self.cov.loc[assets[i], assets[j]]
                #-- Skip over dominated portfolios
                for R,V in pairs:
                    if (R > portfolio_E_Return) & (V < portfolio_E_Variance):
                        next_i = True
                        break
                if next_i:
                    break
                #-- Add the mean/variance pairs to a list for plotting
                pairs.append([portfolio_E_Return, portfolio_E_Variance])
                weights_list.append(weights)
                tickers_list.append(assets)
                self.tickers_list=tickers_list
                self.weights_list =weights_list
                self.pairs = pairs
                break
                
        #-- Plot the risk vs. return of randomly generated portfolios
        #-- Convert the list from before into an array for easy plotting
        mean_variance_pairs = np.array(pairs)
        risk_free_rate=0 
        #-- Include risk free rate her
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mean_variance_pairs[:,1]**0.5, y=mean_variance_pairs[:,0], 
        marker=dict(color=(mean_variance_pairs[:,0]-risk_free_rate)/(mean_variance_pairs[:,1]**0.5), 
        showscale=True, 
        size=7,
        line=dict(width=1),
        colorscale="RdBu",
        colorbar=dict(title="Sharpe<br>Ratio")
        ), 
        mode='markers',
        text=[str(np.array(tickers_list[i])) + "<br>" + str(np.array(weights_list[i]).round(2)) for i in range(len(tickers_list))]))
        fig.update_layout(template='plotly_white',
        xaxis=dict(title='Annualised Risk (Volatility)'),
        yaxis=dict(title='Annualised Return'),
        title='Sample of Random Portfolios',
        width=850,
        height=500)
        fig.update_xaxes(range=[0.18, 0.35])
        fig.update_yaxes(range=[0.05,0.29])
        fig.update_layout(coloraxis_colorbar=dict(title="Sharpe Ratio"))
        self.fig = fig