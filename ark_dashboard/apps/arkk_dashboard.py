import warnings

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
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
SESSION = requests_cache.CachedSession(cache_name='cache', backend='sqlite', expire_after=dt.timedelta(hours=1))

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

def ticker_selector(df, display=True):
    if display:
        display = 'block'
    else:
        display = 'none'

    if not df.empty:
        df = df[['ticker', 'company']].sort_values('ticker').drop_duplicates()
    
    dropdown = dcc.Dropdown(
        id='ticker-selector',
        options=[
            {
                'label': row['ticker'], 
                'value': row['ticker'], 
                'title': row['company']
            } for idx, row in df.dropna().iterrows()
        ],
        value=[],
        multi=True,
        placeholder='Find your favorite stocks...',
        style={'display': display}
    )

    return dropdown

tabs = dcc.Tabs(id='graph-tabs', value='cur-holdings', children=[dcc.Tab(label=y['alias'], value=x) for x, y in TAB_LABELS.items()])

# Main layout of page
layout = html.Div(children=[
    html.Button(id='refresh-holdings', children=[
        html.I(className="fas fa-sync-alt")
    ]),
    dbc.Tooltip(
        "Refresh data",
        target="refresh-holdings"
    ),
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
                options=[{'label': x, 'value': x} for x in np.append(FUNDS, 'All')],
                value="ARKK",
                searchable=False
            )
        ])
    ]),
    dbc.Button("Open modal", id="open-modal"),
    dbc.Modal(
        [
            dbc.ModalHeader("No stock selected", id="selected-stock-name"),
            dcc.Store(id='selected-stock-ticker'),
            dcc.Loading(
            dcc.Graph(
                id='historical-data-graph-ark',
                config={'modeBarButtonsToRemove': 
                    ['toImage', 'zoomIn2d', 'zoomOut2d', 'hoverClosestCartesian', 'resetScale2d', 'toggleSpikelines', 'hoverCompareCartesian', 'hoverClosestCartesian']
                }
            )),
            dbc.ModalBody(id="selected-stock-content"),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-modal", className="ml-auto")
            ),
        ],
        id="stock-analyzer-modal",
        size='xl'
    ),
    tabs,
    html.Div(id='main-graph-container', children=[
        html.Div(id='main-options-container', children=[
            html.Span(id='calendars-div', children=[
                date_range_picker(False),
                date_picker(False),
            ]),
            html.Span(id='ticker-selector-div', children=[
                ticker_selector(pd.DataFrame())
            ])
        ]),
        dcc.Loading(type="cube", children=[
            dcc.Graph(id='main-graph')
        ])
    ]),
    *holdings_store
])

        


##### MAIN METHODS #####

# Create daily transactional plots
def create_daily_transactions_plots(df):

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[str(x) for x in range(min(df.shape[0], 7))],
        column_widths=[0.5, 0.5],
        row_heights=[0.5, 0.5, 0.5, 0.5],
        specs=[[{"colspan": 2}, None],
            [{},             {}],
            [{},             {}],
            [{},             {}]]
    )
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=1000, showlegend=False)

    iternum = 0
    for date, idx in zip(sorted(df['date'].unique(), reverse=True)[:7], [(1,1), (2,1), (2,2), (3,1), (3,2), (4,1), (4,2)]):
        temp = df[df['date'] == date]
        fig.layout.annotations[iternum]['text'] = pd.to_datetime(str(date)).strftime("%A, %b %d")
        fig.add_trace(go.Scatter(
            x=temp['ticker'], 
            y=temp['transaction_value'],
            text=temp['shares'],
            mode='markers', name=str(date)),
            row=idx[0], col=idx[1]
        )
        iternum+=1

    return fig

# Create transaction analyzer plot
def create_transaction_analyzer_plot(df):
    
    fig = go.Figure(layout={'height': 600})
    fig.add_trace(go.Scatter(
                            x=df['ticker'], 
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
    fig.update_layout(
        title={
            'text': "Holdings",
            'x': 0.5,
            'xanchor': "center"
        },
        xaxis_title="Company",
        yaxis_title="Portfolio Allocation (%)",
    )

    return fig




##### CALLBACKS #####

# TRIGGER: Fund dropdown is changed
# OUTPUTS: Updates the displayed fund name
@app.callback(
    Output('fund-name', 'children'),
    [Input('fund-dropdown', 'value')]
)
def change_fund_header(fund):
    if not fund:
        return no_update
    return fund

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
    Output('calendars-div', 'children'),
    [Input('graph-tabs', 'value')]
)
def update_calendar_div(tabname):

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
    [Output('main-graph', 'figure'), Output('ticker-selector-div', 'children')],
    [Input('graph-tabs', 'value'), Input('fund-name', 'children'),
     Input('date-range-picker', 'start_date'), Input('date-range-picker', 'end_date'), 
     Input('date-picker', 'date'), Input('ticker-selector', 'value'), *[Input(x, 'data') for x in FUND_IDS]]
)
def change_graph(tabname, fundname, start, end, curdate, selectorvalues, *funds):

    # Initialize callback context
    cb_ctxt = dash.callback_context.triggered
    trigger = cb_ctxt[0]['prop_id']
    print("Callback params: ", tabname, fundname, start, end, curdate, selectorvalues)

    if not tabname or not fundname:
        raise PreventUpdate

    # Prevent update if callback trigger is a tab change to prevent duplicate trigger activations
    #   This trigger also triggers another callback that will in turn trigger this callback again
    if trigger == 'graph-tabs.value' and tabname in ['trans-view', 'trans-daily']:
        raise PreventUpdate

    # Create initial dataframe based on fund selection and format
    df = create_holdings_from_fundname(fundname, *funds)
    df['date'] = pd.to_datetime(df['date'])
    df['shares'], df['market value($)'], df['weight(%)'] = map(lambda x: x.astype('float64'), [df['shares'], df['market value($)'], df['weight(%)']])

    # Sleep so that loading animation can be seen :)
    import time
    time.sleep(1)

    # Check that holdings data exists
    if df.empty:
        print("Dataframe is empty")
        raise PreventUpdate

    display_ticker_selector = True
    # If the trigger event was a modification of the ticker selection dropdown, filter df accordingly
    if trigger == 'ticker-selector.value' and len(selectorvalues) != 0:
        df = df[df['ticker'].isin(selectorvalues)]

    # Display the POI holdings of the selected fund
    if tabname == "cur-holdings":
        display_ticker_selector = False # This tab does not want to display the ticker selector
        df = df[df['date'] == curdate].sort_values('weight(%)', ascending=False)
        fig = create_holdings_graph(df)
    # Display total transaction value over selected time period
    elif tabname == "trans-view":
        df = df[df['date'].between(start, end)]

        df = create_transaction_log(df)
        df = df.reset_index().groupby(['ticker', 'company']).sum().reset_index()
        df = df.sort_values('transaction_value', ascending=False)

        fig = create_transaction_analyzer_plot(df)
    # Display transactions for previous 7 trading days
    elif tabname == "trans-daily":
        dates = sorted(df['date'].unique(), reverse=True)[:8]
        df = df[df['date'].isin(dates)]
        print(dates)

        # Create a pseudo record of a holdings period indicating 0 shares held before initial purchase so transaction log can be created successfuly
        if len(dates) < 7:
            for idx, row in enumerate(df.drop_duplicates('ticker').itertuples(index=False)):
                df.loc[idx*-1] = [row.company, dates[-1]-np.timedelta64(1, 'D'), row.fund, row.ticker, row.cusip, 0, 0, 0]
            df = df.sort_index()

        df = create_transaction_log(df)
        df = df.reset_index().groupby(['ticker', 'company', 'date']).sum().reset_index()
        df = df.sort_values('transaction_value', ascending=False)

        fig = create_daily_transactions_plots(df)

    # If the trigger event was a modification of the ticker selection dropdown, do not update the ticker selection dropdown
    if trigger == 'ticker-selector.value':
        return fig, no_update
    else:
        return fig, ticker_selector(df)

# TRIGGER: When the open or close modal button is clicked on
# OUTPUTS: Opening or closing the modal
@app.callback(
    Output("stock-analyzer-modal", "is_open"),
    [Input("open-modal", "n_clicks"), Input("close-modal", "n_clicks")],
    [State("stock-analyzer-modal", "is_open")],
)
def toggle_stock_analyzer(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# TRIGGER: When a data point on the main graph is selected
# OUTPUTS: Update the selected ticker dcc.Store and open the stock analyzer modal by simulating a click
@app.callback(
    [Output('open-modal', 'n_clicks'), Output('selected-stock-ticker', 'data')],
    [Input('main-graph', 'clickData')],
    [State('open-modal', 'n_clicks')]
)
def open_stock_analyzer(clickData, modalclicks):

    if not clickData:
        raise PreventUpdate
    if not modalclicks:
        modalclicks = 0

    ticker = clickData['points'][0]['x']

    return modalclicks+1, ticker

# TRIGGER: When the selected stock ticker is updated
# OUTPUTS: Stock analyzer modal is updated with a new historical data graph and other stock metadata
@app.callback(
    [Output('historical-data-graph-ark', 'figure'), Output('selected-stock-name', 'children')],
    [Input('selected-stock-ticker', 'data')],
    [State('fund-name', 'children'), State('open-modal', 'n_clicks'), *[State(x, 'data') for x in FUND_IDS]]
)
def create_historical_plot(ticker, fundname, modalclicks, *funds):

    if not ticker:
        raise PreventUpdate
    if not modalclicks:
        modalclicks = 0
    print(ticker, fundname)

    # Create initial dataframe based on fund selection
    df = create_holdings_from_fundname(fundname, *funds)

    # Filter only the selected ticker
    #df = df[df['ticker'] == ticker]

    # Create transaction log for selected company
    start = time.time()
    df = create_transaction_log(df)
    df = df.loc[ticker]
    print("Transaction log took %s seconds to create" % (time.time()- start))

    # Cast as df if a single transaction is found
    if type(df) == pd.core.series.Series:
        df = df.to_frame().T

    # Ensure columns are appropriately typed
    df['date'] = pd.to_datetime(df['date'])
    df['transaction_value'] = df['transaction_value'].astype('int64')
    df['shares'] = df['shares'].astype('int64')

    # Retrieve the historical data for stock
    historical_data = get_historical_data(ticker, session=SESSION, start_date="2020-01-01")

    # Calculate the price and direction of transactions
    df['price'] = historical_data.loc[df['date']]['Close'].values
    df['direction'] = ['buy' if x else 'sell' for x in df['shares'] > 0]

    # Create graph that overlaps transactions with historical data
    historical_price_fig = impose_event_on_historical(df, historical_data=historical_data, initialize_with_zoom=90)

    return historical_price_fig, df['company'].iloc[0]


def create_holdings_from_fundname(fundname, *funds):

    # Create dataframe based on fund selection
    if fundname == "All":
        df = pd.concat([pd.DataFrame(x) for x in funds])
        df = df.groupby(['company', 'ticker', 'date']).sum().reset_index()
        df['weight(%)'] = df['weight(%)'] / len(funds)
    else:
        idx = FUNDS.index(fundname)
        df = pd.DataFrame(funds[idx])

    return df



########                                    EVENT OVERLAY ON HISTORICAL DATA                                     ########

def impose_event_on_historical(events, historical_data, initialize_with_zoom=0):

    # Scale the transaction value to create relative values that will be used for marker sizing in plotting functions
    # TODO: Update this so that bubbles have a minimum size
    transaction_value_abs = events['transaction_value'].abs()
    max_size = transaction_value_abs.max()
    transaction_value_abs = transaction_value_abs / (max_size/30)

    # Map transaction direction to marker color
    direction_mapper = ['green' if x == 'buy' else 'red' for x in events['direction']]

    # Format the transaction values and shares as more interpretable strings
    transaction_value_format = (events['transaction_value']/1000).round(0).astype('int64').apply(lambda x: "${:,}".format(x))
    shares_format = events['shares'].apply(lambda x: "{:,}".format(x))

    # Plotting
    historical_price_fig = go.Figure(layout={'height': 600})
    historical_price_fig.add_trace(go.Scatter(
                                            x=historical_data.index, 
                                            y=historical_data['Close'],
                                            hovertemplate =
                                            '<b>Stock Price</b>'+
                                            '<br></br>'+
                                            '<i>Date</i>:   %{x}'+
                                            '<br><i>Close</i>: $%{y:.2f}</br>'+
                                            '<extra></extra>',
                                            mode='lines', name="Stock Price"))
    historical_price_fig.add_trace(go.Scatter(
                                            x=events['date'], 
                                            y=events['price'],
                                            mode='markers', name="Transaction",
                                            text=transaction_value_format,
                                            customdata=shares_format,
                                            hovertemplate =
                                            '<b>Transaction</b>'+
                                            '<br></br>'+
                                            '<i>Date</i>:                      %{x}'+
                                            '<i><br>Transaction value</i>: %{text}K<br>'+
                                            '<i>Shares</i>:                  %{customdata}'+
                                            '<extra></extra>',
                                            marker_color=direction_mapper,
                                            marker=dict(size=transaction_value_abs)))
    
    # Initialize the graph with a zoom in on the previous 'initialize_with_zoom' days
    if initialize_with_zoom:
        recent_df = historical_data.loc[pd.to_datetime(dt.date.today() - dt.timedelta(days=initialize_with_zoom)):]
        plot_range = [recent_df['Close'].min(), recent_df['Close'].max()]
        range_span = plot_range[1] - plot_range[0]
        plot_range[0] = plot_range[0] - .2 * range_span
        plot_range[1] = plot_range[1] + .2 * range_span
        historical_price_fig.update_yaxes(
            range=plot_range
        )
        historical_price_fig.update_xaxes(
            range=[dt.date.today() - dt.timedelta(days=initialize_with_zoom), dt.date.today()]
        )

    # Hide specified dates on x axis
    # Configure x axis spike
    historical_price_fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]), # hide weekends
            dict(values=["2020-12-25", "2020-01-01"])  # hide Christmas and New Year's
        ],
        showspikes=True, spikethickness=1, spikecolor='black', spikesnap='cursor', spikemode='across'
    )

    # Configure hover options
    historical_price_fig.update_layout(hovermode='x', spikedistance=100000, hoverdistance=2)

    return historical_price_fig

def get_historical_data(ticker, session=None, start_date="2010-01-01"):

    # Create ticker object
    company = yf.Ticker(ticker)

    # Create a new cached session if one does not exist
    if not session:
        session = requests_cache.CachedSession(cache_name='cache', backend='sqlite', expire_after=dt.timedelta(days=1))

    # Request historical data from yahoo finance
    historical_data = pdr.get_data_yahoo(ticker, start=start_date, end=dt.datetime.now().strftime('%Y-%m-%d'), session=session)
    historical_data.index = pd.to_datetime(historical_data.index)

    return historical_data



########                                    TRANSACTION LOG CREATION                                     ########

def create_transaction_log(df, max_processes=5):

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
    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        logs = executor.map(unwrap_transaction_log_args, [(df[df['date']==x], df[df['date']==y]) for x,y in date_tuples])
    df = pd.concat(list(logs))
    print("Finished creating transaction log: %s seconds" % (time.time()-start))

    return df

def unwrap_transaction_log_args(data):
    return create_transaction_log_single(data[0], data[1])

### Create a transaction log between two holding periods ###
def create_transaction_log_single(prev, cur):
    """
    Create a log of transactions given two dataframes containing daily holdings information

    Returns a log detailing each transaction made of the form
        [ticker shares transaction_value date company]

    Note: Direction of difference is cur-prev, so cur should hold most recent holdings information

    Parameters
    ----------
    prev : dataframe containing information on the initial holdings
    cur : dataframe containing information on the updated holdings
    """

    if cur.empty:
        return pd.DataFrame()
    if prev.empty:
        return 
    
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