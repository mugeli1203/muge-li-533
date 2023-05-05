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
import math
import warnings
warnings.filterwarnings("ignore")
import pandas_market_calendars as mcal
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from dash import Dash, html
import dash_bootstrap_components as dbc

nyse = mcal.get_calendar('NYSE')

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
                start_date = date(2020, 1, 1),
                end_date = date(2023, 3, 23)
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
                    html.Img(src='assets/reactive.png', style={'width': '100%', 'height': '100%'}),# Put your reactive graph here as an image!
                    md = 4
                )
            ],
            align="center",
        ),
        html.H2('Trade Blotter:'),
        dash_table.DataTable(id = "blotter"),
        html.Div(dash_table.DataTable(id="raw"), style={"display": "none"}),
        html.H2('Ledger:'),
        dash_table.DataTable(id = "ledger"),
        html.H2('Behavior of Trading Algorithm'),
        html.Img(src='assets/heat.png', style={'width': '30%', 'height': '30%'}),
        dash_table.DataTable(id = "behavior"),
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

@app.callback(
    Output("ledger", "data"),
    [Input("run-query2", "n_clicks"),Input("blotter", "data")],
    prevent_initial_call=True
)
def ledger(n_clicks, blotter):
    def cal_trading_days(start_dt, end_dt, df):
        start_idx = df[(df['date'] == start_dt) & (df['trip'] == 'ENTER') & (df['status'] == 'SUBMITTED')].index[0]
        end_idx = df[(df['date'] == end_dt) & (df['trip'] == 'ENTER') & (df['status'] == 'SUBMITTED')].index[0]
        return end_idx - start_idx + 1
    raw_data = pd.DataFrame(blotter)
    raw_data = raw_data.set_index('trade_id')
    live_entry_index = raw_data[(raw_data['status'] == 'LIVE') & (raw_data['trip'] == 'EXIT')].index.unique().tolist()
    live_entry_groups = raw_data.loc[live_entry_index]
    live_entry_df = live_entry_groups.groupby(live_entry_groups.index).apply(lambda g: pd.DataFrame({
        "trade_id": g.index[0],
        "asset": g["asset"].iloc[0],
        "dt_enter": g["date"].iloc[0],
        "dt_exit": g["date"].iloc[-1],
        "success": '',
        "n": '',
        "rtn": ''
    }, index=[0]))

    live_not_entry_index = raw_data[
        (raw_data['status'] == 'LIVE') & (raw_data['trip'] == 'ENTER')].index.unique().tolist()
    live_not_entry_groups = raw_data.loc[live_not_entry_index]
    live_not_entry_df = live_not_entry_groups.groupby(live_not_entry_groups.index).apply(lambda g: pd.DataFrame({
        "trade_id": g.index[0],
        "asset": g["asset"].iloc[0],
        "dt_enter": g["date"].iloc[0],
        "dt_exit": '',
        "success": '',
        "n": '',
        "rtn": ''
    }, index=[0]))

    not_entry_index = raw_data[
        (raw_data['status'] == 'CANCELLED') & (raw_data['trip'] == 'ENTER')].index.unique().tolist()
    not_entry_groups = raw_data.loc[not_entry_index]
    not_entry_df = not_entry_groups.groupby(not_entry_groups.index).apply(lambda g: pd.DataFrame({
        "trade_id": g.index[0],
        "asset": g["asset"].iloc[0],
        "dt_enter": g["date"].iloc[0],
        "dt_exit": '',
        "success": 0,
        "n": cal_trading_days(g["date"].iloc[0], g["date"].iloc[-1], raw_data),
        # len(nyse.schedule(start_date=g["date"].iloc[0], end_date=g["date"].iloc[-1])),
        "rtn": ''
    }, index=[0]))

    success_index = raw_data[(raw_data['status'] == 'FILLED') & (raw_data['trip'] == 'EXIT') & (
                raw_data['type'] == 'LMT')].index.unique().tolist()
    success_groups = raw_data.loc[success_index]
    success_df = success_groups.groupby(success_groups.index).apply(lambda g: pd.DataFrame({
        "trade_id": g.index[0],
        "asset": g["asset"].iloc[0],
        "dt_enter": g["date"].iloc[0],
        "dt_exit": g["date"].iloc[-1],
        "success": 1,
        "n": cal_trading_days(g["date"].iloc[0], g["date"].iloc[-1], raw_data),
        # len(nyse.schedule(start_date=g["date"].iloc[0], end_date=g["date"].iloc[-1])),
        "rtn": (math.log(g['price'].iloc[-1] / g['price'].iloc[1])) / cal_trading_days(g["date"].iloc[0],
                                                                                       g["date"].iloc[-1], raw_data)
        # / len(nyse.schedule(start_date=g["date"].iloc[0], end_date=g["date"].iloc[-1]))
    }, index=[0]))

    market_index = raw_data[(raw_data['type'] == 'MKT')].index.unique().tolist()
    market_groups = raw_data.loc[market_index]
    market_df = market_groups.groupby(market_groups.index).apply(lambda g: pd.DataFrame({
        "trade_id": g.index[0],
        "asset": g["asset"].iloc[0],
        "dt_enter": g["date"].iloc[0],
        "dt_exit": g["date"].iloc[-1],
        "success": -1,
        "n": cal_trading_days(g["date"].iloc[0], g["date"].iloc[-1], raw_data),
        # len(nyse.schedule(start_date=g["date"].iloc[0], end_date=g["date"].iloc[-1])),
        "rtn": (math.log(g['price'].iloc[-1] / g['price'].iloc[1])) / cal_trading_days(g["date"].iloc[0],
                                                                                       g["date"].iloc[-1], raw_data)
        # / len(nyse.schedule(start_date=g["date"].iloc[0], end_date=g["date"].iloc[-1]))
    }, index=[0]))

    dfs = [live_entry_df, live_not_entry_df, not_entry_df, success_df, market_df]

    merged_df = pd.concat(dfs, ignore_index=True)[
        ['trade_id', 'asset', "dt_enter", "dt_exit", "success", "n", "rtn"]].sort_values('trade_id')
    merged_df.to_csv('ledger.csv')
    return merged_df.to_dict('records')

@app.callback(
    Output("behavior", "data"),
    [Input("run-query2", "n_clicks"), Input("ledger", "data")],
    prevent_initial_call=True
)
def behavior(n_clicks, ledger):
    # define function to fit and predict using a scikit-learn model
    def fit_and_predict(model, parameters, X_train, y_train, X_test):
        # create grid search object to find best model parameters
        grid = GridSearchCV(model, parameters)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        # predict using best model and calculate accuracy score
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return best_model, y_pred, accuracy

    def return_calculate(prices: pd.DataFrame, method="LOG", dateColumn="Dates") -> pd.DataFrame:
        vars = prices.columns.values.tolist()  # list of the column names
        nVars = len(vars)
        vars.remove(dateColumn)  # remove the column of "date"
        if nVars == len(vars):
            raise ValueError(f"dateColumn: {dateColumn} not in DataFrame: {vars}")
        nVars = nVars - 1
        p = np.array(prices.drop(columns=[dateColumn]))
        n = p.shape[0]  # the number of rows
        m = p.shape[1]  # the number of column
        p2 = np.empty((n - 1, m))
        for i in range(n - 1):
            for j in range(m):
                p2[i, j] = p[i + 1, j] / p[i, j]
        if method.upper() == "ARITHMETIC":
            p2 = p2 - 1.0
        elif method.upper() == "LOG":
            p2 = np.log(p2)
        else:
            raise ValueError(f"method: {method} must be in (\"LOG\",\"DISCRETE\")")
        dates = prices[dateColumn][1:]
        out = pd.DataFrame({dateColumn: dates})
        for i in range(nVars):
            out[vars[i]] = p2[:, i]
        return out

    lg = pd.read_csv('ledger.csv')
    lg["Dates"] = pd.to_datetime(lg["dt_enter"])
    raw_features = pd.read_csv('hw4_data.csv')
    raw_features['Dates'] = pd.to_datetime(raw_features['Dates'], format='%Y/%m/%d')
    features_p = raw_features.copy()

    returns = return_calculate(features_p[['Dates', 'IVV US Equity', 'IVV AU Equity']], method="LOG",
                               dateColumn="Dates")
    returns = returns.rename(columns={'IVV US Equity': 'IVV US return', 'IVV AU Equity': 'IVV AU return'})
    features = pd.merge(features_p, returns, on='Dates', how='inner')

    features['Dates'] = features['Dates'].shift(-1)
    features = features.drop(features.index[-1])
    merged_data = pd.merge(features, lg, on='Dates', how='inner')

    merged_data = merged_data.drop(['trade_id', 'asset', 'dt_enter', 'dt_exit', 'n', 'rtn', 'IVV US return'], axis=1)
    merged_data = merged_data.set_index('Dates')
    success = merged_data.pop('success')
    merged_data.insert(0, 'success', success)
    merged_data = merged_data.dropna()

    data = merged_data
    X = data.iloc[:, 3:-1]
    print(X)
    y = data.iloc[:, 0]

    # normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split data into training, testing, and prediction sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # create PCA object and apply dimensionality reduction
    pca = PCA(n_components=5)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # define dictionary of scikit-learn models and their corresponding parameters for grid search
    models = {'SVM': (SVC(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'sigmoid']}),
              'Random Forest': (RandomForestClassifier(), {'n_estimators': [10, 50, 100], 'max_depth': [5, 10, 20]}),
              'Gradient Boosting': (GradientBoostingClassifier(),
                                    {'learning_rate': [0.01, 0.1, 1], 'n_estimators': [50, 100, 150],
                                     'max_depth': [3, 5, 7]}),
              'Logistic Regression': (LogisticRegression(), {'C': [0.1, 1, 10]}),
              'Naive Bayes': (GaussianNB(), {}),
              'Multi-layer Perceptron': (MLPClassifier(), {'hidden_layer_sizes': [(100,), (50, 50), (20, 20, 20)],
                                                           'activation': ['tanh', 'relu'],
                                                           'alpha': [0.0001, 0.001, 0.01]}),
              'k-Nearest Neighbors': (
                  KNeighborsClassifier(), {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']})
              }

    def calculate_alpha_beta(model_name):
        # fit model and predict on selected_data
        datap = pd.read_csv(f'{model_name}_predictions.csv')
        datap.columns = ['Dates', 'pridiction']  # 修改列名
        datap["Dates"] = pd.to_datetime(datap["Dates"])

        # 将数据按index合并
        merged_tem = pd.merge(datap, lg, on='Dates', how='inner')

        # 筛选出相同index下x参数为-1和1的数据，并且y参数为1
        selected_data = merged_tem[(merged_tem['success'] == -1) | (merged_tem['success'] == 1)]
        selected_data = selected_data[selected_data['pridiction'] == 1]

        bm_rtn = []
        for index, row in selected_data.iterrows():
            enter_dt = pd.to_datetime(row['dt_enter'])
            exit_dt = pd.to_datetime(row['dt_exit'])
            n = len(nyse.schedule(start_date=enter_dt, end_date=exit_dt))
            enter_p = raw_features.loc[raw_features['Dates'] == enter_dt, 'IVV US Equity'].iloc[0]
            exit_p = raw_features.loc[raw_features['Dates'] == exit_dt, 'IVV US Equity'].iloc[0]
            r = np.log(exit_p / enter_p) / n
            bm_rtn.append(r)
        selected_data["bm_rtn"] = bm_rtn

        # 将r作为因变量，return作为自变量进行回归分析
        model = LinearRegression()
        model.fit(selected_data[['bm_rtn']], selected_data['rtn'])
        alpha = model.intercept_
        beta = model.coef_[0]

        return alpha, beta

    def hoeffding_inequality(p, n, delta):
        return 2 * np.exp(-2 * n * (delta ** 2) * (p ** 2))

    def check_model_information_loss(y_true, y_pred, delta=0.05):
        n_samples = len(y_true)
        n_classes = len(np.unique(y_true))
        class_probs = np.array([np.mean(y_true == i) for i in range(n_classes)])
        error_limits = np.array([hoeffding_inequality(p, n_samples, delta) for p in class_probs])
        pred_probs = np.array([np.mean(y_pred == i) for i in range(n_classes)])

        return np.all(pred_probs <= class_probs + error_limits)

    # create list of model names and corresponding alpha and beta values
    results = []
    for name, (model, parameters) in models.items():
        best_model, y_pred, accuracy = fit_and_predict(model, parameters, X_train_pca, y_train, X_test_pca)
        merged = pd.concat([y_test, pd.Series(y_pred, index=y_test.index)], axis=1)
        merged.drop(merged.columns[0], axis=1, inplace=True)
        merged.to_csv(f'{name}_predictions.csv')
        alpha, beta = calculate_alpha_beta(name)
        is_info_loss = check_model_information_loss(y_test, y_pred)
        results.append((name, accuracy, alpha, beta, is_info_loss))

    df = pd.DataFrame(results, columns=['Model Name', 'Accuracy', 'Alpha', 'Beta', 'Is Information Loss'])
    print(df)
    return df.to_dict('records')


if __name__ == '__main__':
    app.run_server(debug=True)


