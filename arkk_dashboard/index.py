import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from apps import arkk_dashboard

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

app.validation_layout = html.Div([
    app.layout,
    arkk_dashboard.layout
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return arkk_dashboard.layout
    else:
        return '404' + str(pathname)

if __name__ == '__main__':
    app.run_server(debug=True)