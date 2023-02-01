import os
from datetime import datetime

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import refinitiv.dataplatform.eikon as ek
from dash import Dash, html, dcc, dash_table, Input, Output, State

ek.set_app_key(os.getenv('EIKON_API'))

dt_prc_div_splt = pd.read_csv('unadjusted_price_history.csv')

app = Dash(__name__)
app.layout = html.Div([
    html.Div([
        dbc.Label("Benchmark", size="md"),
        dcc.Input(id='benchmark-id', type='text', value="IVV"),
        dbc.Label('Asset', size='md'),
        dcc.Input(id='asset-id', type='text', value="AAPL.O")
    ]),
    html.Div([
        dcc.DatePickerRange(
            id='date_range_id',
            min_date_allowed=datetime(1900, 1, 1),
            max_date_allowed=datetime.now(),
            start_date=datetime(2017, 1, 1),
            end_date=datetime.now()
        )
    ]),
    html.Button('QUERY Refinitiv', id='run-query', n_clicks=0),
    html.H2('Raw Data from Refinitiv'),
    dash_table.DataTable(
        id="history-tbl",
        page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}
    ),
    html.H2('Historical Returns'),
    dash_table.DataTable(
        id="returns-tbl",
        page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}
    ),
    html.H2('Alpha & Beta Scatter Plot'),
    html.Div([
        dcc.DatePickerRange(
            id='pd_range_id',
            min_date_allowed=datetime(1900, 1, 1),
            max_date_allowed=datetime.now(),
            start_date=datetime(2017, 1, 1),
            end_date=datetime.now()
        )
    ]),
    html.Button("Plot", id='run-plot', n_clicks=0),
    dcc.Graph(id="ab-plot"),
    html.P(id='summary-text', children="")
])


@app.callback(
    Output("history-tbl", "data"),
    Input("run-query", "n_clicks"),
    [State("date_range_id", "start_date"), State("date_range_id", "end_date"), State('benchmark-id', 'value'),
     State('asset-id', 'value')],
    prevent_initial_call=True
)
def query_refinitiv(n_clicks, start_date, end_date, benchmark_id, asset_id):
    assets = [benchmark_id, asset_id]
    start_date_object = datetime.fromisoformat(start_date)
    start_date_string = start_date_object.strftime("%Y-%m-%d")
    end_date_object = datetime.fromisoformat(end_date)
    end_date_string = end_date_object.strftime("%Y-%m-%d")
    prices, prc_err = ek.get_data(
        instruments=assets,
        fields=[
            'TR.OPENPRICE(Adjusted=0)',
            'TR.HIGHPRICE(Adjusted=0)',
            'TR.LOWPRICE(Adjusted=0)',
            'TR.CLOSEPRICE(Adjusted=0)',
            'TR.PriceCloseDate'
        ],
        parameters={
            'SDate': start_date_string,
            'EDate': end_date_string,
            'Frq': 'D'
        }
    )

    divs, div_err = ek.get_data(
        instruments=assets,
        fields=[
            'TR.DivExDate',
            'TR.DivUnadjustedGross',
            'TR.DivType',
            'TR.DivPaymentType'
        ],
        parameters={
            'SDate': start_date_string,
            'EDate': end_date_string,
            'Frq': 'D'
        }
    )

    splits, splits_err = ek.get_data(
        instruments=assets,
        fields=['TR.CAEffectiveDate', 'TR.CAAdjustmentFactor'],
        parameters={
            "CAEventType": "SSP",
            'SDate': start_date_string,
            'EDate': end_date_string,
            'Frq': 'D'
        }
    )

    prices.rename(
        columns={
            'Open Price': 'open',
            'High Price': 'high',
            'Low Price': 'low',
            'Close Price': 'close'
        },
        inplace=True
    )
    prices.dropna(inplace=True)
    prices['Date'] = pd.to_datetime(prices['Date']).dt.date

    divs.rename(
        columns={
            'Dividend Ex Date': 'Date',
            'Gross Dividend Amount': 'div_amt',
            'Dividend Type': 'div_type',
            'Dividend Payment Type': 'pay_type'
        },
        inplace=True
    )
    divs.dropna(inplace=True)
    divs['Date'] = pd.to_datetime(divs['Date']).dt.date
    divs = divs[(divs.Date.notnull()) & (divs.div_amt > 0)]

    splits.rename(
        columns={
            'Capital Change Effective Date': 'Date',
            'Adjustment Factor': 'split_rto'
        },
        inplace=True
    )
    splits.dropna(inplace=True)
    splits['Date'] = pd.to_datetime(splits['Date']).dt.date

    unadjusted_price_history = pd.merge(
        prices, divs[['Instrument', 'Date', 'div_amt']],
        how='outer',
        on=['Date', 'Instrument']
    )
    unadjusted_price_history['div_amt'].fillna(0, inplace=True)

    unadjusted_price_history = pd.merge(
        unadjusted_price_history, splits,
        how='outer',
        on=['Date', 'Instrument']
    )
    unadjusted_price_history['split_rto'].fillna(1, inplace=True)

    if unadjusted_price_history.isnull().values.any():
        raise Exception('missing values detected!')

    return (unadjusted_price_history.to_dict('records'))


@app.callback(
    Output("returns-tbl", "data"),
    Input("history-tbl", "data"),
    prevent_initial_call=True
)
def calculate_returns(history_tbl):
    dt_prc_div_splt = pd.DataFrame(history_tbl)

    # Define what columns contain the Identifier, date, price, div, & split info
    ins_col = 'Instrument'
    dte_col = 'Date'
    prc_col = 'close'
    div_col = 'div_amt'
    spt_col = 'split_rto'

    dt_prc_div_splt[dte_col] = pd.to_datetime(dt_prc_div_splt[dte_col])
    dt_prc_div_splt = dt_prc_div_splt.sort_values([ins_col, dte_col])[
        [ins_col, dte_col, prc_col, div_col, spt_col]].groupby(ins_col)
    numerator = dt_prc_div_splt[[dte_col, ins_col, prc_col, div_col]].tail(-1)
    denominator = dt_prc_div_splt[[prc_col, spt_col]].head(-1)

    return (
        pd.DataFrame({
            'Date': numerator[dte_col].reset_index(drop=True),
            'Instrument': numerator[ins_col].reset_index(drop=True),
            'rtn': np.log(
                (numerator[prc_col] + numerator[div_col]).reset_index(drop=True) / (
                        denominator[prc_col] * denominator[spt_col]
                ).reset_index(drop=True)
            )
        }).pivot_table(
            values='rtn', index='Date', columns='Instrument'
        ).reset_index().to_dict('records')
    )


@app.callback(
    Output("ab-plot", "figure"),
    Input("run-plot", "n_clicks"),
    Input("returns-tbl", "data"),
    [State("pd_range_id", "start_date"), State("pd_range_id", "end_date"), State('benchmark-id', 'value'),
     State('asset-id', 'value')],
    prevent_initial_call=True
)
def render_ab_plot(n_cicks, returns, start_date, end_date, benchmark_id, asset_id):
    start_date_object = datetime.fromisoformat(start_date)
    start_date_string = start_date_object.strftime("%Y-%m-%d")
    end_date_object = datetime.fromisoformat(end_date)
    end_date_string = end_date_object.strftime("%Y-%m-%d")
    new_rtn = []
    for i in range(len(returns)):
        if start_date_string <= returns[i]['Date'] <= end_date_string:
            returns[i].pop('Date')
            new_rtn.append(returns[i])
    return (
        px.scatter(new_rtn, x=benchmark_id, y=asset_id, trendline='ols')
    )


@app.callback(
    Output("summary-text", "children"),
    Input("ab-plot", "figure"),
    prevent_initial_call=True
)
def cal_ab(figure):
    x = figure['data'][0]["x"]
    y = figure['data'][0]["y"]
    df = pd.DataFrame({'X': x, 'Y': y})
    model = px.get_trendline_results(px.scatter(df, x="X", y="Y", trendline="ols"))
    results = model.iloc[0]["px_fit_results"]
    alpha = results.params[0]
    beta = results.params[1]
    return html.Div(f'Alpha is {alpha.round(5)}, Beta is {beta.round(5)}')


if __name__ == '__main__':
    app.run_server(debug=True)
