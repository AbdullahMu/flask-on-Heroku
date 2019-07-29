
# coding: utf-8

# In[ ]:


import flask
import numpy as np
import pickle
from flask import render_template


app = flask.Flask(__name__)

with open('model.pkl', 'rb') as picklefile:
    model = pickle.load(picklefile)
    
#http://127.0.0.1:5000/
#http://127.0.0.1:5000/page
@app.route('/')
@app.route('/page')
def page():
    with open("templates/page.html", 'r') as viz_file:
        return viz_file.read()

#http://127.0.0.1:5000/result
@app.route('/result', methods=['POST', 'GET'])
def result():
    '''Gets prediction using the HTML form'''
    if flask.request.method == 'POST':

        inputs = flask.request.form

        pclass = inputs['pclass'][0]
        sex = inputs['sex'][0]
        age = inputs['age'][0]
        fare = inputs['fare'][0]
        sibsp = inputs['sibsp'][0]

        item = np.array([pclass, sex, age, fare, sibsp])
        item = item.reshape(1, len(item))
        score = model.predict_proba(item)
        results = {'survival chances': score[0,1], 'death chances': score[0,0]}
        return flask.jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)


