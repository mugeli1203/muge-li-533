from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime, date, timedelta
import plotly.express as px
import pandas as pd
from datetime import datetime
import numpy as np
import os
import refinitiv.dataplatform.eikon as ek
import refinitiv.data as rd

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
ek.set_app_key(os.getenv('EIKON_API'))

percentage = dash_table.FormatTemplate.percentage(3)

controls = dbc.Card(
    [
        dbc.Row(html.Button('QUERY Refinitiv', id='run-query', n_clicks=0)),
        dbc.Row([
            html.H5('Asset:',
                    style={'display': 'inline-block', 'margin-right': 20}),
            dcc.Input(id='asset', type='text', value="IVV",
                      style={'display': 'inline-block',
                             'border': '1px solid black'}),
            dbc.Table(
                [
                    html.Thead(html.Tr([html.Th("α1"), html.Th("n1")])),
                    html.Tbody([
                        html.Tr([
                            html.Td(
                                dbc.Input(
                                    id='alpha1',
                                    type='number',
                                    value=-0.01,
                                    max=1,
                                    min=-1,
                                    step=0.01
                                )
                            ),
                            html.Td(
                                dcc.Input(
                                    id='n1',
                                    type='number',
                                    value=3,
                                    min=1,
                                    step=1
                                )
                            )
                        ])
                    ])
                ],
                bordered=True
            ),
            dbc.Table(
                [
                    html.Thead(html.Tr([html.Th("α2"), html.Th("n2")])),
                    html.Tbody([
                        html.Tr([
                            html.Td(
                                dbc.Input(
                                    id='alpha2',
                                    type='number',
                                    value=0.01,
                                    max=1,
                                    min=-1,
                                    step=0.01
                                )
                            ),
                            html.Td(
                                dcc.Input(
                                    id='n2',
                                    type='number',
                                    value=5,
                                    min=1,
                                    step=1
                                )
                            )
                        ])
                    ])
                ],
                bordered=True
            )
        ]),
        dbc.Row([
            dcc.DatePickerRange(
                id='refinitiv-date-range',
                min_date_allowed = date(2015, 1, 1),
                max_date_allowed = datetime.now(),
                start_date = date(2023, 1, 30),
                end_date = date(2023, 2, 9)
            )
        ]),
        dbc.Row(html.H5('If you change n and alpha, please click submit to modify',
                    style={'display': 'inline-block', 'margin-right': 20})),
        dbc.Row(html.Button('Submit', id='run-query2', n_clicks=0)),
        dbc.Row(html.H5('Page created by Muge Li, Meijing Hou and Yiyang Huang',
                    style={'display': 'inline-block', 'margin-right': 20})),
    ],
    body=True
)
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(controls, md=4),
                dbc.Col(
                    html.Img(src='/assets/reactive.png', style={'width': '100%', 'height': '100%'}),# Put your reactive graph here as an image!
                    md = 4
                )
            ],
            align="center",
        ),
        html.H2('Trade Blotter:'),
        dash_table.DataTable(id = "blotter"),
        html.Div(dash_table.DataTable(id="raw"), style={"display": "none"}),
        #dbc.Row([
            #html.H2('blotter'),
            #dash_table.DataTable(
                #id="blotter",
                #page_action='none',
                #style_table={'height': '300px', 'overflowY': 'auto'}
            #),
        #]),
    ],
    fluid=True
)

@app.callback(
    Output("raw", "data"),
    Input("run-query", "n_clicks"),
    [State('asset', 'value'),State('refinitiv-date-range', 'start_date'),
     State('refinitiv-date-range', 'end_date')],
    prevent_initial_call=True
)
def query_refinitiv(n_clicks, asset, start_date, end_date):
    assets = [asset]
    ivv_prc, ivv_prc_err = ek.get_data(
        instruments=assets,
        fields=[
            'TR.OPENPRICE(Adjusted=0)',
            'TR.HIGHPRICE(Adjusted=0)',
            'TR.LOWPRICE(Adjusted=0)',
            'TR.CLOSEPRICE(Adjusted=0)',
            'TR.PriceCloseDate'
        ],
        parameters={
            'SDate': start_date,
            'EDate': end_date,
            'Frq': 'D'
        }
    )
    ivv_prc['Date'] = pd.to_datetime(ivv_prc['Date']).dt.date
    ivv_prc.drop(columns='Instrument', inplace=True)
    return(ivv_prc.to_dict('records'))

@app.callback(
    Output("blotter", "data"),
    [Input("run-query2", "n_clicks"),Input("raw", "data")],
    [State('asset', 'value'),State("alpha1","value"),State("n1","value"),
     State("alpha2","value"),State("n2","value")],
    prevent_initial_call=True
)
def query_refinitiv2(n_clicks, raw, asset, alpha1,n1,alpha2,n2):
    ivv_prc = pd.DataFrame(raw)
    ivv_prc['Date'] = list(pd.to_datetime(ivv_prc["Date"].iloc[:]).dt.date)
    rd.open_session()

    next_business_day = rd.dates_and_calendars.add_periods(
        start_date=ivv_prc['Date'].iloc[-1].strftime("%Y-%m-%d"),
        period="1D",
        calendars=["USA"],
        date_moving_convention="NextBusinessDay",
    )
    rd.close_session()
    # submitted entry orders
    submitted_entry_orders = pd.DataFrame({
        "trade_id": range(1, ivv_prc.shape[0]),
        "date": list(pd.to_datetime(ivv_prc["Date"].iloc[1:]).dt.date),
        "asset": asset,
        "trip": 'ENTER',
        "action": "BUY",
        "type": "LMT",
        "price": round(
            ivv_prc['Close Price'].iloc[:-1] * (1 + alpha1),
            2
        ),
        'status': 'SUBMITTED'
    })

    # if the lowest traded price is still higher than the price you bid, then the
    # order never filled and was cancelled.
    with np.errstate(invalid='ignore'):
        cancelled_entry_orders = submitted_entry_orders[
            np.greater(
                ivv_prc['Low Price'].iloc[1:][::-1].rolling(3).min()[::-1].to_numpy(),
                submitted_entry_orders['price'].to_numpy()
            )
        ].copy()
    cancelled_entry_orders.reset_index(drop=True, inplace=True)
    cancelled_entry_orders['status'] = 'CANCELLED'

    #将日期换成三天之后
    cancelled_entry_orders['date'] = pd.DataFrame(
        {'cancel_date': submitted_entry_orders['date'].iloc[(n1-1):].to_numpy()},
        index=submitted_entry_orders['date'].iloc[:(1-n1)].to_numpy()
    ).loc[cancelled_entry_orders['date']]['cancel_date'].to_list()
    #print(cancelled_entry_orders)

    filled_entry_orders = submitted_entry_orders[
        submitted_entry_orders['trade_id'].isin(
            list(
                set(submitted_entry_orders['trade_id']) - set(
                    cancelled_entry_orders['trade_id']
                )
            )
        )
    ].copy()
    filled_entry_orders.reset_index(drop=True, inplace=True)
    filled_entry_orders['status'] = 'FILLED'
    for i in range(0, len(filled_entry_orders)):

        idx1 = np.flatnonzero(
            ivv_prc['Date'] == filled_entry_orders['date'].iloc[i]
        )[0]

        ivv_slice = ivv_prc.iloc[idx1:(idx1+n1)]['Low Price']

        fill_inds = ivv_slice <= filled_entry_orders['price'].iloc[i]

        if (len(fill_inds) < n1) & (not any(fill_inds)):
            filled_entry_orders.at[i, 'status'] = 'LIVE'
        else:
            filled_entry_orders.at[i, 'date'] = ivv_prc['Date'].iloc[
                fill_inds.idxmax()
            ]

    live_entry_orders = pd.DataFrame({
        "trade_id": ivv_prc.shape[0],
        "date": pd.to_datetime(next_business_day).date(),
        "asset": asset,
        "trip": 'ENTER',
        "action": "BUY",
        "type": "LMT",
        "price": round(ivv_prc['Close Price'].iloc[-1] * (1 + alpha1), 2),
        'status': 'LIVE'
    },
        index=[0]
    )

    if any(filled_entry_orders['status'] =='LIVE'):
        live_entry_orders = pd.concat([
            filled_entry_orders[filled_entry_orders['status'] == 'LIVE'],
            live_entry_orders
        ])
        live_entry_orders['date'] = pd.to_datetime(next_business_day).date()

    filled_entry_orders = filled_entry_orders[
        filled_entry_orders['status'] == 'FILLED'
        ]

    entry_orders = pd.concat(
        [
            submitted_entry_orders,
            cancelled_entry_orders,
            filled_entry_orders,
            live_entry_orders
        ]
    ).sort_values(["date", 'trade_id'])

###following are for exit:
    submitted_exit_orders = pd.DataFrame({
        "trade_id": filled_entry_orders['trade_id'],
        "date": list(pd.to_datetime(filled_entry_orders["date"]).dt.date),
        "asset": asset,
        "trip": 'EXIT',
        "action": "SELL",
        "type": "LMT",
        "price": round(
            filled_entry_orders['price'] * (1 + alpha2),
            2
        ),
        'status': 'SUBMITTED'
    })

    submitted_exit_orders = submitted_exit_orders.reset_index(drop=True)
    #print(submitted_exit_orders)

    diff_exit_orders = submitted_exit_orders.copy()
    for i in range(0, len(submitted_exit_orders)):

        idx1 = np.flatnonzero(
            ivv_prc['Date'] == submitted_exit_orders['date'].iloc[i]
        )[0]

        ivv_slice = np.append(ivv_prc.iloc[(idx1+1):(idx1+n2)]['High Price'].to_numpy(), ivv_prc.iloc[idx1]['Close Price'])
        #print(len(ivv_slice))
        if submitted_exit_orders['price'].iloc[i] >= ivv_slice.max():
            if len(ivv_slice) == n2:
                diff_exit_orders.at[i, 'status'] = 'CANCELLED'
            else:
                diff_exit_orders.at[i, 'status'] = 'LIVE'
        else:
            diff_exit_orders.at[i, 'status'] = 'FILLED'

    #print(diff_exit_orders)

    filled_exit_orders = diff_exit_orders.loc[diff_exit_orders['status'] == 'FILLED']
    filled_exit_orders = pd.DataFrame(filled_exit_orders)
    #print(filled_exit_orders)
    cancelled_exit_orders = diff_exit_orders.loc[diff_exit_orders['status'] == 'CANCELLED']
    cancelled_exit_orders = pd.DataFrame(cancelled_exit_orders)
    #print(cancelled_exit_orders)
    live_exit_orders = diff_exit_orders.loc[diff_exit_orders['status'] == 'LIVE']
    live_exit_orders = pd.DataFrame(live_exit_orders)
    #print(live_exit_orders)


    cancelled_exit_orders['date'] = pd.DataFrame(
        {'cancel_date': ivv_prc['Date'].iloc[(n2-1):].to_numpy()},
        index=ivv_prc['Date'].iloc[:(1-n2)].to_numpy()
    ).loc[cancelled_exit_orders['date']]['cancel_date'].to_list()
    #print(cancelled_exit_orders)
    # 将第一个DataFrame的日期列名改为 date
    ivv_prcs = ivv_prc.rename(columns={'Date': 'date'})

    merge_cancel = pd.merge(ivv_prcs, cancelled_exit_orders, on='date')
    #print(merge_cancel)
    market_exit_orders = pd.DataFrame({#All arrays must be of the same length
        "trade_id": list(cancelled_exit_orders['trade_id']),
        "date": list(pd.to_datetime(cancelled_exit_orders["date"].iloc[:]).dt.date),
        "asset": asset,
        "trip": 'EXIT',
        "action": "SELL",
        "type": "MKT",
        "price": list(merge_cancel['Close Price']),
        'status': 'FILLED'
    })
    filled_exit_orders = filled_exit_orders.reset_index(drop=True)
    for i in range(0, len(filled_exit_orders)):
        idx1 = np.flatnonzero(
            ivv_prc['Date'] == filled_exit_orders['date'].iloc[i]
        )[0]
        #print(ivv_prc['Date'])
        ivv_slice = np.append(ivv_prc.iloc[idx1]['Close Price'], ivv_prc.iloc[(idx1+1):(idx1+n2)]['High Price'].to_numpy())
        ivv_slice = pd.Series(ivv_slice)
        #print(ivv_slice)

        fill_inds = ivv_slice >= filled_exit_orders['price'].iloc[i]

        if (len(fill_inds) < n1) & (not any(fill_inds)):
            filled_exit_orders.at[i, 'status'] = 'LIVE'
        else:
            filled_exit_orders.at[i, 'date'] = ivv_prc['Date'].iloc[idx1 + fill_inds.idxmax()]
    #print(filled_exit_orders)

    exit_orders = pd.concat(
        [
            submitted_exit_orders,
            cancelled_exit_orders,
            filled_exit_orders,
            live_exit_orders,
            market_exit_orders
        ]
    ).sort_values(["date", 'trade_id'])
    ##print('exit orders:')
    ##print(exit_orders)


    blotter = pd.concat(
        [
            entry_orders,
            exit_orders
        ]
    ).sort_values([ 'trade_id','date'])

    return blotter.to_dict('records')










if __name__ == '__main__':
    app.run_server(debug=True)


