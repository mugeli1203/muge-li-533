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
import pylab
from dash import Dash, html
import dash_bootstrap_components as dbc

nyse = mcal.get_calendar('NYSE')


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


# read data and split into features (X) and target (y)
# ledger = pd.read_csv('ledger.csv')
# ledger["Dates"] = pd.to_datetime(ledger["dt_enter"])
# raw_features = pd.read_csv('hw4_data.csv')
# raw_features['Dates'] = pd.to_datetime(raw_features['Dates'], format='%Y/%m/%d')
# features = pd.read_csv('hw4_data.csv')
# features['Dates'] = pd.to_datetime(features['Dates'], format='%Y/%m/%d')
# features['Dates'] = features['Dates'].shift(-1)
# features = features.drop(features.index[-1])
# merged_data = pd.merge(features, ledger, on='Dates', how='inner')


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


ledger = pd.read_csv('ledger.csv')
ledger["Dates"] = pd.to_datetime(ledger["dt_enter"])
raw_features = pd.read_csv('hw4_data.csv')
raw_features['Dates'] = pd.to_datetime(raw_features['Dates'], format='%Y/%m/%d')
features_p = raw_features.copy()

returns = return_calculate(features_p[['Dates', 'IVV US Equity', 'IVV AU Equity']], method="LOG", dateColumn="Dates")
returns = returns.rename(columns={'IVV US Equity': 'IVV US return', 'IVV AU Equity': 'IVV AU return'})
features = pd.merge(features_p, returns, on='Dates', how='inner')

features['Dates'] = features['Dates'].shift(-1)
features = features.drop(features.index[-1])
merged_data = pd.merge(features, ledger, on='Dates', how='inner')

merged_data = merged_data.drop(['trade_id', 'asset', 'dt_enter', 'dt_exit', 'n', 'rtn', 'IVV US return'], axis=1)
merged_data = merged_data.set_index('Dates')
success = merged_data.pop('success')
merged_data.insert(0, 'success', success)
merged_data = merged_data.dropna()

data = merged_data
X = data.iloc[:, 3:]
y = data.iloc[:, 0]
# sns = pd.concat([y, X], axis=1)
corr=X.corr()
#设置画布
#Plot figsize
pylab.mpl.rcParams['font.sans-serif'] = ['SimHei']
pylab.mpl.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(10, 10))
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(corr, cmap='RdBu', annot=True, fmt=".2f")
#Apply xticks
plt.xticks(range(len(corr.columns)), corr.columns)
#Apply yticks
plt.yticks(range(len(corr.columns)), corr.columns)
#show plot
plt.show()

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
                                                       'activation': ['tanh', 'relu'], 'alpha': [0.0001, 0.001, 0.01]}),
          'k-Nearest Neighbors': (
          KNeighborsClassifier(), {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']})
          }


def calculate_alpha_beta(model_name):
    # fit model and predict on selected_data
    datap = pd.read_csv(f'{model_name}_predictions.csv')
    datap.columns = ['Dates', 'pridiction']  # 修改列名
    datap["Dates"] = pd.to_datetime(datap["Dates"])

    # 将数据按index合并
    merged_tem = pd.merge(datap, ledger, on='Dates', how='inner')

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

# create Dash app
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# create table to display alpha and beta values
table = dbc.Table(
    [
        html.Thead(html.Tr([html.Th("Model"), html.Th("Accuracy"), html.Th("Alpha"), html.Th("Beta"), html.Th("Information Loss")])),
        html.Tbody([html.Tr([html.Td(name), html.Td(accuracy), html.Td(alpha), html.Td(beta), html.Td(str(is_info_loss))]) for
                    name, accuracy, alpha, beta, is_info_loss in results]),
    ],
    bordered=True,
    hover=True,
    responsive=True,
    striped=True,
)

# create app layout
app.layout = dbc.Container(
    [
        dbc.Row(dbc.Col(html.H1("Alpha and Beta Values for Machine Learning Models"), width=12)),
        dbc.Row(dbc.Col(table, width=12)),
    ],
    fluid=True,
)

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
