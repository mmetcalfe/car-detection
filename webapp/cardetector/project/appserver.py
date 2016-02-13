import flask
from flask import render_template
# from flask import request
from flask import jsonify

app = flask.Flask(__name__)

@app.route('/')
def index():
    return render_template('hello.html')

@app.route('/hello')
def hello():
    return render_template('hello.html')

@app.route('/_add_numbers', methods=['POST'])
def add_numbers():
    """Add two numbers server side, ridiculous but well..."""
    a = flask.request.args.get('a', 0, type=int)
    b = flask.request.args.get('b', 0, type=int)
    return jsonify(result=a + b)

if __name__ == '__main__':
    app.run(debug=True)
