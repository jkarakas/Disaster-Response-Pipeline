import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from sklearn.externals import joblib
from sqlalchemy import create_engine
import plotly.graph_objs as goj


app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages_categories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


def create_first_plot(df):
    """Create a plotly figure of a messages per category barplot

    Parameters
    ----------
    df : pandas.Dataframe
        The dataset

    Returns
    -------
    fig:
        The plotly figure
    top_cat;
        The top 10 categories
    """

    # Grop by categories
    categories = df.iloc[:, 4:].sum().sort_values(ascending=False)

    color_bar = 'Teal'

    data = [goj.Bar(
        x=categories.index,
        y=categories,
        marker=dict(color=color_bar),
        opacity=0.8
    )]

    layout = goj.Layout(
        title="Messages per Category",
        xaxis=dict(
            title='Categories',
            tickangle=45
        ),
        yaxis=dict(
            title='# of Messages',
            tickfont=dict(
                color='DarkGreen'
            )
        )
    )

    return goj.Figure(data=data, layout=layout), categories.index[:10]


def create_second_plot(df, top_cat):
    """Create a plotly figure of a messages per category barplot

    Parameters
    ----------
    df : pandas.Dataframe
        The dataset

    Returns
    -------
    fig:
        The plotly figure
    """

    # Grop by categories
    genres = df.groupby('genre').sum()[top_cat]

    color_bar = 'DarkGreen'

    data = []
    for cat in genres.columns[1:]:
        data.append(goj.Bar(
                    x=genres.index,
                    y=genres[cat],
                    name=cat)
                    )

    layout = goj.Layout(
        title="Categories per genre (Top 10)",
        xaxis=dict(
            title='Genres',
            tickangle=45
        ),
        yaxis=dict(
            title='# of Messages per Caegorie',
            tickfont=dict(
                color=color_bar
            )
        ),
        barmode='stack'
    )

    return goj.Figure(data=data, layout=layout)


# get figures and top categories
fig1, top_cat = create_first_plot(df)
fig2 = create_second_plot(df, top_cat)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # encode plotly graphs in JSON
    graphs = [fig1, fig2]
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
