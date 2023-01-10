from Elements import Portfolio,Stock

if __name__ == '__main__':
    portfolio = Portfolio(['AAPL','BND','VB','VEA','VOO'])
    start = '2013-01-01'
    end = '2018-03-01'
    timeframe = 'daily'
    portfolio.makeStocks(start,end,timeframe)
    portfolio.calculateReturns([0.2,0.3,0.1,0.1,0.3])