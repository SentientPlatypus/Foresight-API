from flaskApp.stockApi import app
from flask_cors import CORS, cross_origin
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
    app.config['CORS_HEADERS'] = 'Content-Type'
    cors = CORS(app)
    app.run()