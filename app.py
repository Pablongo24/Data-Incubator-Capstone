from flask import Flask, render_template, request, url_for
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8
import pickle
from os import path, getcwd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import pandas as pd

app = Flask(__name__)

# Directories:
MODELDIR = path.join(getcwd(), 'models')

# TODO: change to Revelio's model (atm using model from TDI-final presentation)
with open(path.join(MODELDIR, 'TDI-XGB_model.pkl'), 'rb') as f:
    model = pickle.load(f)

# These columns will change once I update the model to Revelio's model
d = defaultdict(LabelEncoder)
cols_to_transf = ['employer', 'job title', 'state', 'city', 'case status']


def encode_future_data(df, cols_to_transf):
    df_transf = df[cols_to_transf]
    df_non_transf = df.drop(cols_to_transf, axis=1)

    fit = df_transf.apply(lambda x: d[x.name].transform(x))

    df = pd.concat([fit, df_non_transf], axis=1, join='outer')
    return df


def process_request(predict_dict, cols_to_transf, future_data_column_order):
    predict_df = pd.DataFrame.from_dict(predict_dict)
    predict_df_transf = predict_df[cols_to_transf]
    predict_df_non_transf = predict_df.drop(cols_to_transf, axis=1)

    for col in predict_df_transf.columns:
        predict_df_transf[col] = predict_df_transf[col].str.upper()

    predict_df = pd.concat([predict_df_transf, predict_df_non_transf], axis=1, join='outer')

    predict_df = encode_future_data(predict_df, cols_to_transf)

    predict_df = predict_df[future_data_column_order]
    return predict_df


# TODO: create a predict_dict from the requests.post


@app.route('/', methods=['GET', 'POST'])
def index():

    # Static resources:
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    # Render results if available:
    html = render_template('index.html',
                           js_resources=js_resources,
                           css_resources=css_resources,
                           companies=['Google', 'Amazon', 'Netflix'],
                           locations=['New York City', 'San Francisco', 'Washington, D.C.'],
                           job_titles=['Data Scientist', 'Software Engineer', 'Product Manager']
                           )

    return encode_utf8(html)


if __name__ == '__main__':
    app.run(debug=True, use_debugger=True, use_reloader=True, passthrough_errors=False, port=33507)
