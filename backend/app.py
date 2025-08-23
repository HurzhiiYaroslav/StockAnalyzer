import os
import sys
from flask import Flask
from flask_cors import CORS

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.services import ml_utils 
from .routes import api
from .ml_inference import predictor

def create_app():
    """Application factory to create and configure the Flask app."""
    app = Flask(__name__)
    CORS(app)

    if not predictor.get_latest_model_run_id():
        print("="*60)
        print("WARNING: Could not find any trained model in the models directory.")
        print("The prediction API will not be available.")
        print("Please train a model first.")
        print("="*60)


    app.register_blueprint(api, url_prefix='/api')
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)