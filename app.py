from flask import Flask, render_template, request, url_for
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8
import pickle
from os import path, getcwd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import xgboost as xgb
import numpy as np

app = Flask(__name__)

# Directories:
MODELDIR = path.join(getcwd(), 'models')

# Load lists of job titles, locations, and employers to populate the app's dropdown menus and make predictions:
with open(path.join(MODELDIR, 'locations-list.txt'), 'r') as f:
    locations = f.readlines()

with open(path.join(MODELDIR, 'job-titles-list.txt'), 'r') as f:
    job_titles = f.readlines()

with open(path.join(MODELDIR, 'employers-list.txt'), 'r') as f:
    employers = f.readlines()

# TODO: change to Revelio's model (atm using model from TDI-final presentation)
with open(path.join(MODELDIR, 'TDI-XGB_model.pkl'), 'rb') as f:
    model = pickle.load(f)

# These columns will change once I update the model to Revelio's model
d = defaultdict(LabelEncoder)
cols_transf = ['employer', 'job title', 'state', 'city']

for col in cols_transf:
    d[col] = LabelEncoder()
    d[col].classes_ = np.load(path.join(MODELDIR, '{}.npy'.format(col).replace(' ', '_')))


def encode_future_data(df, cols_to_transf):
    df_transf = df[cols_to_transf]
    df_non_transf = df.drop(cols_to_transf, axis=1)

    for k in d.keys():
        print(k)

    fit = df_transf.apply(lambda x: d[x.name].transform(x))

    df = pd.concat([fit, df_non_transf], axis=1, join='outer')
    return df


def process_request(predict_dict, cols_to_transf, future_data_column_order):
    predict_df = pd.DataFrame.from_dict(predict_dict)
    predict_df_transf = predict_df[cols_to_transf]
    predict_df_non_transf = predict_df.drop(cols_to_transf, axis=1)

    for cols in predict_df_transf.columns:
        predict_df_transf[col] = predict_df_transf[cols].str.upper()

    predict_df = pd.concat([predict_df_transf, predict_df_non_transf], axis=1, join='outer')

    predict_df = encode_future_data(predict_df, cols_to_transf)

    predict_df = predict_df[future_data_column_order]
    return predict_df


@app.route('/', methods=['GET', 'POST'])
def index():

    # Static resources:
    return_str = ''
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    if request.method == 'POST':
        job_title = request.form.get('job_title')
        location = request.form.get('location')
        employer = request.form.get('employer')

        city = location.split(',')[0].strip()
        state = location.split(',')[1].strip()

        # 'employer', 'job title', 'state', 'city', 'case status'

        predict_dict = {'employer': [employer], 'job title': [job_title], 'state': [state], 'city': [city],
                        'submit year': [2018]}

        cols_to_transf = ['employer', 'job title', 'state', 'city']
        future_data_column_order = ['employer', 'job title', 'state', 'city', 'submit year']

        predict_df = process_request(predict_dict, cols_to_transf, future_data_column_order)
        d_new_data = xgb.DMatrix(predict_df)
        new_predictions = model.predict(d_new_data)
        return_str = 'Your predicted salary as {} at {} in {}, {} is ${:,.0f}'.format(job_title, employer, city, state,
                                                                                      int(np.expm1(new_predictions)))

    # Render results if available:
    html = render_template('index.html',
                           _anchor="news",
                           js_resources=js_resources,
                           css_resources=css_resources,
                           return_str=return_str,
                           companies=employers,
                           locations=locations,
                           job_titles=job_titles,
                           )

    return encode_utf8(html)


if __name__ == '__main__':
    app.run(debug=True, use_debugger=True, use_reloader=True, passthrough_errors=False, port=33507)
