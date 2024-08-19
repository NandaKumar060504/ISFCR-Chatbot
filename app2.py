from flask import Flask, render_template, request, jsonify
from app1 import llm_chain


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    message = request.form['msg']
    return llm_chain(message)



if __name__ == '__main__':
    app.run() 