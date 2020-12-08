import warnings

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.dash import no_update
from dash.exceptions import PreventUpdate

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import plotly
    import plotly.express as px
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots

import pandas as pd
import os
import numpy as np
import time
import datetime as dt
from concurrent.futures import ProcessPoolExecutor
import requests_cache
import requests
from pymongo import MongoClient

import yfinance as yf
from pandas_datareader import data as pdr

from app import app




##### GLOBALS #####

# Create new cached session
SESSION = requests_cache.CachedSession(cache_name='cache', backend='sqlite', expire_after=dt.timedelta(days=1))

TAB_LABELS = {
    "summary": {"alias": "Summary"},
    "cur-holdings": {"alias": "Current Holdings"},
    "trans-view": {"alias": "Transaction Viewer"},
    "trans-daily": {"alias": "Transactions (previous 7 days)"}
}

# Store list of ARK fund names
FUNDS = ['ARKK', 'ARKW', 'ARKQ', 'ARKF', 'ARKG']
FUND_IDS = ['%s_store' % x for x in FUNDS]
holdings_store = [dcc.Store(id=x) for x in FUND_IDS]



##### LAYOUT COMPONENTS #####

# Date selection to control 
def date_range_picker(display=True, start=dt.date.today() - dt.timedelta(days=7), end=dt.date.today()):
    if display:
        display = 'block'
    else:
        display = 'none'
    date_range_picker = dcc.DatePickerRange(
        id='date-range-picker',
        min_date_allowed=dt.date(2020, 10, 6),
        start_date=start,
        end_date=end,
        with_portal=True,
        day_size=75,
        style={'display': display}
    )
    return date_range_picker

# Date selection to control 
def date_picker(display=True, start=dt.date.today()):
    if display:
        display = 'block'
    else:
        display = 'none'
    date_picker = dcc.DatePickerSingle(
        id='date-picker',
        min_date_allowed=dt.date(2020, 10, 6),
        date=start,
        with_portal=True,
        day_size=75,
        style={'display': display}
    )
    return date_picker

tabs = dcc.Tabs(id='graph-tabs', value='cur-holdings', children=[dcc.Tab(label=y['alias'], value=x) for x, y in TAB_LABELS.items()])

# Main layout of page
layout = html.Div(children=[
    html.Div(id="fund-header", children=[
        html.Div(id="fund-display", children=[
            html.Span("You are viewing the "),
            html.Span(id="fund-name"),
            html.Span(" fund")
        ]),
        html.Div(className='break'),
        html.Div(id='fund-select-container', children=[
            html.Div("Select a new fund: "),
            dcc.Dropdown(
                id='fund-dropdown',
                options=[{'label': x, 'value': x} for x in FUNDS],
                value="ARKK",
                searchable=False
            )
        ])
    ]),
    tabs,
    html.Div(id='main-graph-container', children=[
        html.Div(id='main-div', children=[
            date_range_picker(False),
            date_picker(False)
        ]),
        html.Button('Refresh', id='refresh-holdings'),
        dcc.Graph(id='main-graph')
    ]),
    html.Div(id='historical-data-wrap', children=[
        html.Div(children='Historical price graph'),
        dcc.Graph(id='historical-data-graph-ark')
    ], style={'display': 'none'}),
    *holdings_store
])
        


##### MAIN METHODS #####

# Create daily transactional plots
def create_daily_transactions_plots(df):

    fig = make_subplots(
        rows=4, cols=2,
        column_widths=[0.5, 0.5],
        row_heights=[0.5, 0.5, 0.5, 0.5],
        specs=[[{"colspan": 2}, None],
            [{},             {}],
            [{},             {}],
            [{},             {}]]
    )
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=1000)

    for date, idx in zip(sorted(df['date'].unique(), reverse=True)[:7], [(1,1), (2,1), (2,2), (3,1), (3,2), (4,1), (4,2)]):
        temp = df[df['date'] == date]
        fig.add_trace(go.Scatter(
            x=temp['ticker'], 
            y=temp['transaction_value'],
            text=temp['shares'],
            mode='markers', name=str(date)),
            row=idx[0], col=idx[1]
        )

    return fig

# Create transaction analyzer plot
def create_transaction_analyzer_plot(df):
    
    fig = go.Figure(layout={'height': 600})
    fig.add_trace(go.Scatter(
                            x=df.index, 
                            y=df['transaction_value'],
                            text=df['shares'],
                            mode='markers', name="All Transactions",
                            marker=dict(size=10)))
    fig.update_xaxes(tickangle=45)

    return fig

# Create the current holdings plot
def create_holdings_graph(df):
    
    fig = go.Figure(layout={'height': 600})
    fig.add_trace(go.Scatter(
                            x=df['ticker'], 
                            y=df['weight(%)'],
                            text=[(x,y) for x,y in zip(df['market value($)'], df['shares'])],
                            mode='markers', name="All Transactions",
                            marker=dict(size=10)))
    fig.update_xaxes(tickangle=45)

    return fig




##### CALLBACKS #####

# TRIGGER: Fund dropdown is changed
# OUTPUTS: Updates the displayed fund name
@app.callback(
    Output('fund-name', 'children'),
    [Input('fund-dropdown', 'value')]
)
def change_fund_header(fund):
    return fund

def retrieve_holding(fundname):
    df = holdings_store

@app.callback(
    [Output(x, 'data') for x in FUND_IDS],
    [Input('refresh-holdings', 'n_clicks')]
)
def update_holdings_data(nclicks):

    server = "InvestmentDB"
    username = os.environ['MONGO_ATLAS_USERNAME']
    password = os.environ['MONGO_ATLAS_PASSWORD']

    client = MongoClient("mongodb+srv://%s:%s@%s.ccy3i.mongodb.net/%s?retryWrites=true&w=majority" % (username, password, server.lower(), server))
    db = client['ARK_fund']

    fundata = []
    for fund in FUNDS:
        collection = db["%s_Holdings" % fund]
        data = pd.DataFrame(collection.find({}))
        data.drop('_id', inplace=True, axis=1)
        data = data.to_dict('records')
        fundata.append(data)

    return fundata

@app.callback(
    Output('main-div', 'children'),
    [Input('graph-tabs', 'value')]
)
def update_helper_div(tabname):

    if tabname == "cur-holdings":
        return date_range_picker(False), date_picker(True)
    elif tabname == "trans-view":
        return date_range_picker(True), date_picker(False)
    elif tabname == "trans-daily":
        return date_range_picker(False), date_picker(False)
    else:
        return no_update
    


# TRIGGER: Tab selection is changed, Fund display name is changed
# OUTPUTS: Content of main container / graph
@app.callback(
    Output('main-graph', 'figure'),
    [Input('graph-tabs', 'value'), Input('fund-name', 'children'), 
     Input('date-range-picker', 'start_date'), Input('date-range-picker', 'end_date'), 
     Input('date-picker', 'date'), *[Input(x, 'data') for x in FUND_IDS]]
)
def change_graph(tabname, fundname, start, end, curdate, *funds):

    cb_ctxt = dash.callback_context.triggered
    print(tabname, fundname, start, end, curdate)
    if not tabname or not fundname:
        return no_update, no_update

    # Prevent update if callback trigger is a tab change to prevent duplicate trigger activations
    #   This trigger also triggers another callback that will in turn trigger this callback again
    if cb_ctxt[0]['prop_id'] == 'graph-tabs.value' and tabname in ['trans-view', 'trans-daily']:
        raise PreventUpdate

    # Create initial dataframe based on fund selection
    if fundname == "All":
        df = pd.concat([pd.DataFrame(x) for x in funds])
        df = df.groupby(['ticker', 'date']).sum().reset_index()
    else:
        idx = FUNDS.index(fundname)
        df = pd.DataFrame(funds[idx])
    df['date'] = pd.to_datetime(df['date'])
    df['shares'], df['market value($)'], df['weight(%)'] = map(lambda x: x.astype('float64'), [df['shares'], df['market value($)'], df['weight(%)']])

    if df.empty:
        print("Dataframe is empty")
        raise PreventUpdate

    # Display the current holdings of the selected fund
    if tabname == "cur-holdings":
        df = df[df['date'] == curdate].sort_values('weight(%)', ascending=False)
        fig = create_holdings_graph(df)

    elif tabname == "trans-view":
        df = df[df['date'].between(start, end)]

        df = create_transaction_log(df)
        df = df.groupby(level=0).sum().sort_values('transaction_value', ascending=False)

        fig = create_transaction_analyzer_plot(df)

    elif tabname == "trans-daily":
        dates = sorted(df['date'].unique(), reverse=True)[:8]
        df = df[df['date'].isin(dates)]

        df = create_transaction_log(df)
        df = df.reset_index().groupby(['ticker', 'date']).sum().reset_index()
        df = df.sort_values('transaction_value', ascending=False)

        fig = create_daily_transactions_plots(df)

    return fig

# TRIGGER: When a ticker on the main graph is clicked on
# OUTPUTS: Historical data graph is updated with data
#@app.callback(
#    [Output('historical-data-graph-ark', 'figure'), Output('historical-data-wrap', 'style')],
#    [Input('main-graph', 'clickData')],
#    [State('fund-name', 'children')]
#)
def create_historical_plot(clickData, fundname):
    print(clickData, fundname)

    if not clickData:
        return no_update, no_update

    if fundname == 'All':
        df = pd.concat([x for x in TRANSACTION_LOG.values()])
        df = df.groupby(['ticker', 'date']).sum().reset_index()
    else:
        df = TRANSACTION_LOG[fundname].copy(deep=True)

    ticker = clickData['points'][0]['x']

    historical_data = get_historical_data(ticker, SESSION, start_date="2020-01-01")

    historical_price_fig = impose_event_on_historical(df, ticker, historical_data=historical_data)

    recent_df = historical_data.loc[pd.to_datetime(dt.date.today() - dt.timedelta(days=30)):]
    plot_range = [recent_df['Close'].min(), recent_df['Close'].max()]
    range_span = plot_range[1] - plot_range[0]
    plot_range[0] = plot_range[0] - .2 * range_span
    plot_range[1] = plot_range[1] + .2 * range_span
    historical_price_fig.update_yaxes(
        range=plot_range
    )

    #historical_price_fig.update_layout(hovermode='x unified')

    return historical_price_fig, {'display': 'block'}



def get_historical_data(ticker, session=None, start_date="2010-01-01"):
    # Query for stock information
    company = yf.Ticker(ticker)

    if not session:
        session = requests_cache.CachedSession(cache_name='cache', backend='sqlite', expire_after=dt.timedelta(days=1))

    historical_data = pdr.get_data_yahoo(ticker, start=start_date, end=dt.datetime.now().strftime('%Y-%m-%d'), session=session)
    historical_data.index = pd.to_datetime(historical_data.index)

    return historical_data

def impose_event_on_historical(df, ticker, session=None, trans_val_div=1000, 
                               date_col='date', price_col='price', shares_col='shares', trans_type_col='type_colored',
                               recent_range=True, historical_data=None):
    if historical_data is None:
        historical_data = get_historical_data(ticker, session, start_date="2020-01-01")

    df = df[df['ticker'] == ticker]
    df['transaction_value'] = (df['transaction_value'] / trans_val_div).astype('int64')
    max_size = df.describe().loc['75%']['transaction_value']

    if price_col not in df.columns:
        df[price_col] = historical_data.loc[df[date_col]]['Close'].values
    if trans_type_col not in df.columns:
        df[trans_type_col] = ['green' if x else 'red' for x in (df[shares_col] > 0)]
    transaction_value_abs = df['transaction_value'].abs()

    transaction_value_abs[transaction_value_abs > max_size] = max_size
    transaction_value_abs = transaction_value_abs / (max_size/50)
    transaction_value_abs[transaction_value_abs < 10] = 10

    trans_val_unit = ''
    if trans_val_div == 1000:
        trans_val_unit = 'k'

    historical_price_fig = go.Figure(layout={'height': 600})
    historical_price_fig.add_trace(go.Scatter(
                                            x=historical_data.index, 
                                            y=historical_data['Close'],
                                            hovertemplate =
                                            '<b>Date</b>: %{x}'+
                                            '<br><b>Close</b>: $%{y:.0f}<br>',
                                            mode='lines', name="Price"))
    historical_price_fig.add_trace(go.Scatter(
                                            x=pd.DatetimeIndex(df[date_col]).date, 
                                            y=df[price_col],
                                            mode='markers', name="Recommendations",
                                            text=df['transaction_value'],
                                            customdata=df[shares_col],
                                            hovertemplate =
                                            '<b>Date</b>: %{x}'+
                                            '<br><b>Transaction value</b>: $%{text:.2f}' + trans_val_unit + '<br>'+
                                            '<b>Shares: %{customdata}</b>',
                                            marker_color=df[trans_type_col],
                                            marker=dict(size=transaction_value_abs)))
    
    if recent_range:
        historical_price_fig.update_xaxes(
            range=[dt.date.today() - dt.timedelta(days=30), dt.date.today()]
        )

    historical_price_fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]), #hide weekends
            dict(values=["2020-12-25", "2020-01-01"])  # hide Christmas and New Year's
        ]
    )

    return historical_price_fig



def create_transaction_log(df):

    # Compile matrix of dates that serve as bounds for each transaction
    dates = sorted(df['date'].unique())
    date_tuples = []
    prev_date = dates[0]
    for idx, date in enumerate(dates[1:], 1):
        date_tuples.append((prev_date, date))
        prev_date = date

    # Create a transaction log for each of previously established time frames and compile into one master log
    start = time.time()
    print("Creating transaction log")
    with ProcessPoolExecutor(max_workers=1) as executor:
        logs = executor.map(unwrap_transaction_log_args, [(df[df['date']==x], df[df['date']==y]) for x,y in date_tuples])
    df = pd.concat(list(logs))
    print("Finished creating transaction log: %s seconds" % (time.time()-start))

    return df

def unwrap_transaction_log_args(data):
    return create_transaction_log_single(data[0], data[1])

### Create a transaction log between two holding periods ###
def create_transaction_log_single(prev, cur):
    
    def format_df(df):
        df = df.copy()
        df.dropna(subset=['ticker'], inplace=True)
        df['ticker'] = df['ticker'].apply(lambda x: x.replace('"', "").split(" ")[0])
        df.set_index('ticker', inplace=True)
        df['pps'] = df['market value($)'] / df['shares']
        return df
    
    # Format and join transaction days
    prev = format_df(prev)
    cur = format_df(cur)
    joined = prev.join(cur, how='outer', lsuffix='_prev', rsuffix='_cur')
    #return joined
    # Compute transaction information from joined df
    transaction_log = pd.DataFrame()
    transaction_log['shares'] = joined['shares_cur'] - joined['shares_prev']
    transaction_log['transaction_value'] = transaction_log['shares'] * joined['pps_cur']
    transaction_log['date'] = joined['date_cur']
    transaction_log['company'] = joined['company_cur']
    
    # Handle companies that are entering or exiting the portfolio
    prev_tickers = set(prev.index)
    cur_tickers = set(cur.index)
    new_purchases = cur_tickers - prev_tickers
    new_sells = prev_tickers - cur_tickers
    for pur in new_purchases:
        transaction_log.loc[pur, 'shares'] = cur.loc[pur, 'shares']
        transaction_log.loc[pur, 'transaction_value'] = cur.loc[pur, 'market value($)']
        transaction_log.loc[pur, 'company'] = cur.loc[pur, 'company']
    for sell in new_sells:
        transaction_log.loc[sell, 'shares'] = -1*prev.loc[sell, 'shares']
        transaction_log.loc[sell, 'transaction_value'] = -1*prev.loc[sell, 'market value($)']
        transaction_log.loc[sell, 'company'] = prev.loc[sell, 'company']
        
    # Format transaction log df
    transaction_log = transaction_log[transaction_log['shares'] != 0]
    transaction_log = transaction_log.sort_values(by=['transaction_value'], ascending=False)
    try:
        transaction_log['date'] = transaction_log['date'].interpolate(method='pad')
    except:
        pass
    
    return transaction_log