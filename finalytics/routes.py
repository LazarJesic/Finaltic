from flask import render_template, request, redirect, Response, url_for, flash
from finalytics.auxiliary import load_stocks
from finalytics.Elements import Stock, spy_stats
from finalytics import app,db , bcrypt
import sys
import json
import plotly
import yfinance as yf
from finalytics.forms import RegistrationForm,LoginForm
from finalytics.models import User, Portfolio, Ticker
from flask_login import login_user, current_user, logout_user, login_required

tickers = list(load_stocks('stocks.txt'))
timeframe = 'daily'

### MAKE THE ROUTES ###

@app.route("/")
@app.route("/home")
def index():
    return render_template("index.html", tickers=tickers)

@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            flash(f'Login "{user.username}" successful.', 'success')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route("/account")
@login_required
def account():
    return render_template('account.html', title='Account')

@app.route("/stock", methods=["POST","GET"])
def stock():
    if request.method == 'GET':
        # Validate name
        ticker = request.args.get("name")
        ticker = ticker.upper()
        if not ticker:
            print('hi', file=sys.stderr)
            return render_template("failure.html", message="Missing name")
        if ticker not in tickers:
            return render_template("failure.html", message="Stock not found")
        fromDate = request.args.get("fromDate")
        toDate = request.args.get("toDate")
        if not fromDate:
            return render_template("failure.html", message="Missing the starting analysis date")
        if not toDate:
            return render_template("failure.html", message="Missing the ending analysis date")
        if fromDate>toDate:
            return render_template("failure.html", message="The ending date is greater than the starting date")
        else:
            stock = Stock(ticker=ticker)
            stock.get_prices(fromDate,toDate,timeframe)
            stock.get_statistics('Returns') 
            tableStats = round(stock.stats,5).to_html(header="true",table_id='stats',index=False )
            tablePrice = round(stock.prices,5).to_html(header="true",table_id='price')
            fig1 = stock.get_candlestick()
            graphJSON1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
            fig2 = stock.get_SPYgraph()
            graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
            SPYStats = spy_stats('Returns',fromDate,toDate,timeframe)
            SPYStats = round(SPYStats ,5).to_html(header="true",table_id='stats',index=False )
            #Download info to display
            info = yf.Ticker(stock.name).info
            nameL = info['longName']
            summary = info['longBusinessSummary']
            companyLogo = info['logo_url']
            returns = stock.prices['Returns']
            corr,summaryCAPM  = stock.get_SPYcorr()
            graphJSON3= json.dumps(corr, cls=plotly.utils.PlotlyJSONEncoder)
            LR =stock.get_CAPM()
            slope, intercept, rvalue, pvalue,stderr = LR
            #plt=returns.plot(kind='kde')
            #plt = makePlotPath(plt)
            weeklyReturn = ((intercept+1)**5-1)*100
            yearlyReturn = ((intercept+1)**251-1)*100
            dailyReturn = intercept*100
            fig4 = stock.test_stationarity()
            graphJSON4 = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
            arima, fig5, r2_arima, mse_arima, mae_arima = stock.ARIMAAll()
            graphJSON5 = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)
            return render_template("success.html",nameL=nameL,companyLogo=companyLogo,summary=summary, tableStats=tableStats,tablePrice=tablePrice , stock = ticker,graphJSON1=graphJSON1, 
            graphJSON2=graphJSON2, SPYStats=SPYStats,tickers=tickers,graphJSON3=graphJSON3, beta=round(slope,5),intercept=round(intercept,5), rvalue=round(rvalue,5),
            weeklyReturn=round(weeklyReturn,5),yearlyReturn=round(yearlyReturn,5), dailyReturn=round(dailyReturn,5),summaryCAPM=summaryCAPM,graphJSON4=graphJSON4,arima=arima, graphJSON5=graphJSON5,
            r2_arima = round(r2_arima,5), mse_arima = round(mse_arima,5), mae_arima = round(mae_arima,5))
    
    elif request.method == 'POST':
        if request.form.get('Submit_portfolio') == "Add to Portfolio":
            name = request.args.get('name').upper()
            tickersDB = (Ticker.query.order_by(Ticker.name).all())
            tickersused = set()
            for ticker in tickersDB:
                tickersused.add(ticker.name) 
            if name not in tickersused:
                ticker = Ticker(name=name)
                db.session.add(ticker)
                db.session.commit()
            else:
                flash('Already in roaster', 'danger')
            return redirect('/portfolio')


@app.route("/portfolio", methods=["POST","GET"])
def portfolio():
    if request.method == 'GET':
        stocks = Ticker.query.order_by(Ticker.name).all()
        infoStocks = []
        for stock in stocks:
            info = yf.Ticker(stock.name).info
            info['tickerName'] = stock.name
            infoStocks.append(info)
        return render_template('portfolio.html', stocks = stocks,infoStocks=infoStocks,tickers=tickers)

    elif request.method == 'POST':
        if request.form.get('Submit_portfolio') == "submit":
            names=request.form.getlist('checkStock')
            name = request.args.get('PortfolioName')
            portfolio = Portfolio(name=name,user_id=current_user)
            db.session.add(portfolio)
            db.session.commit()
            return redirect('/portfolioStatisitcs')


@app.route("/portfolioStatisitcs", methods=["POST","GET"])
def portfolioStatisitcs():
    if request.method == 'GET':
        name = request.args.get('name')
        portfolio = Portfolio(name=name, user_id=current_user)
        db.session.add(portfolio)
        db.session.commit()       
    return render_template('portfolioStatisitcs.html',PortfolioName = portfolio)
