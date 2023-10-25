import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash import dash_table
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, roc_curve, auc
import plotly.graph_objs as go
import dash_bootstrap_components as dbc

#dataset
#change your target folder accordingly with your own dataset
data = pd.read_csv("diabetes.csv") 
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

# Define styles
styles = {
    'report-container': {
        'whiteSpace': 'pre-wrap',
        'overflowX': 'auto',
        'marginTop': '20px'
    },
    'train-button': {
        'width': '20%',
        'height': '40px',
        'marginTop': '20px',
        'textAlign': 'center',
        'margin': 'auto'
    }
}

#layout of the dashboard
app.layout = html.Div([
    html.Div([
        html.H1("Model Simulator", style={"textAlign": "center"})
    ]),

    html.Div([
        html.H4("Data (First 5 Rows)"),
        dash_table.DataTable(
            id='sample-data-table',
            columns=[
                {'name': col, 'id': col} for col in data.columns
            ],
            data=data.head().to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': 'lightgrey',
                'fontWeight': 'bold',
                'fontSize': '14px'
            },
            style_cell={
                'fontSize': '12px',
                'textAlign': 'left',
                'maxWidth': 0,
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            }
        )
    ]),

    html.Div([
        html.H4("Model Selection"),
        dcc.RadioItems(
            id='model-selection',
            options=[
                {'label': 'Random Forest Classifier', 'value': 'rf'},
                {'label': 'Support Vector Machine', 'value': 'svm'},
                {'label': 'Logistic Regression', 'value': 'lr'},
                {'label': 'KNearest Neighbor', 'value': 'knn'},
                {'label': 'Naive Bayes', 'value': 'nb'}
            ],
            value='rf'
        ),
    ]),

    html.Button('Train Model', id='train-button', style=styles['train-button']),

    html.Div([
        html.H4('Classification Report', style={'textAlign': 'center'}),
        html.Div(id='classification-report', style=styles['report-container'])
    ]),

    html.Div([
        html.H4('ROC CURVE', style={'textAlign': 'center'}),
        dcc.Graph(id='roc-curve')
    ]),

    html.Div([
        html.H4('Model Comparison', style={'textAlign': 'center'}),
        dcc.Graph(id='model-comparison-chart')
    ])
])


def update_model_comparison_chart(selected_model):
    models = {
        'Random Forest': RandomForestClassifier(),
        'Support Vector Machine': SVC(probability=True),
        'Logistic Regression': LogisticRegression(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB()
    }

    auc_values = []
    model_names = []
#this bellow  code will run all the model on data and show their AUC Comaprison on bar chart
    for model_name, model in models.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        auc_values.append(roc_auc)
        model_names.append(model_name)

    # bar chart to compare models based on AUC values
    model_comparison_chart = go.Figure()
    model_comparison_chart.add_trace(go.Bar(x=model_names, y=auc_values,
                                            marker_color='skyblue',
                                            text=auc_values,
                                            textposition='outside',
                                            texttemplate='%{text:.3f}'))

    model_comparison_chart.update_layout(title='Model Comparison',
                                         xaxis_title='Model',
                                         yaxis_title='AUC Value',
                                         showlegend=False)

    return model_comparison_chart


# Callback to handle model training and classification report display
@app.callback([Output('classification-report', 'children'),
               Output('roc-curve', 'figure'),
               Output('model-comparison-chart', 'figure')],
              [Input('train-button', 'n_clicks')],
              [Input('model-selection', 'value')])
def train_model(n_clicks, selected_model):
    if n_clicks is None:
        return '', go.Figure(), go.Figure()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Logic for training selected model
    if selected_model == 'rf':
        # Train Random Forest Classifier
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    elif selected_model == 'svm':
        # Train Support Vector Machine
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = SVC(probability=True)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    elif selected_model == 'lr':
        # Train Logistic Regression
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    elif selected_model == 'knn':
        # Train K-Nearest Neighbors
       # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    elif selected_model == 'nb':
        # Train Naive Bayes
      #  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = GaussianNB()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    # Generate classification report
    report = classification_report(y_test, predictions)
    report_output = html.Pre(report, style={'whiteSpace': 'pre-wrap'})

    # Generate ROC curve
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    roc_curve_plot = go.Figure()
    roc_curve_plot.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve (AUC = {:.2f})'.format(roc_auc)))
    roc_curve_plot.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                                 xaxis_title='False Positive Rate',
                                 yaxis_title='True Positive Rate',
                                 showlegend=True)

    # Generate Model Comparison Chart
    model_comparison_chart = update_model_comparison_chart(selected_model)

    return report_output, roc_curve_plot, model_comparison_chart


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
