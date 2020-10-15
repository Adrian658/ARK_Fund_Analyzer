import dash
import dash_bootstrap_components as dbc

external_stylesheets = [
    'https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css'
    ]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=False)
server = app.server