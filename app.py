from flask import Flask, render_template, request, url_for
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8

app = Flask(__name__)


def placeholder_result():
    return None


@app.route('/', methods=['GET', 'POST'])
def index():

    # Static resources:
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

# Render results is available
    html = render_template('index.html',
                           js_resources=js_resources,
                           css_resources=css_resources,
                           companies = ['Google', 'Amazon', 'Netflix'],
                           locations = ['New York City', 'San Francisco', 'Washington, D.C.'],
                           job_titles = ['Data Scientist', 'Software Engineer', 'Product Manager']
                           )

    return encode_utf8(html)


if __name__ == '__main__':
    app.run(debug=True, use_debugger=True, use_reloader=True, passthrough_errors=False, port=33507)
