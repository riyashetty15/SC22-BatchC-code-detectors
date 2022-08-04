# import requirements needed
from flask import Flask, render_template, request, url_for
from utils import get_base_url
from collections import Counter
import pickle
import matplotlib.pyplot as plt
import numpy as np

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 45690
base_url = get_base_url(port)

model = pickle.load(open('model.pkl','rb'))

# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')

# Open the model


@app.route(f'{base_url}')
def home():
    return render_template('index.html')

@app.route(f"{base_url}", methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return(flask.render_template('index.html', predict = ""))
    if request.method == 'POST':
        inp_features = [float(val) for key, val in request.form.items()]
        input_variables = np.append(inp_features,1)
        input_variables = input_variables.reshape(1,-1)
        prediction = model.predict(input_variables)[0]
        print(prediction)
        return render_template('index.html', predict= int(prediction))
                                     

if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'cocalc7.ai-camp.dev/'
    
    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)
