"""Flask application factory. Startup verifies all artifacts; requests only infer."""
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge
from .service import InferenceService

ROOT = Path(__file__).resolve().parents[1]

def create_app(root=ROOT):
    service = InferenceService(root)
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=32768, DEBUG=False)
    app.extensions['inference_service'] = service

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.get('/api/bootstrap')
    def bootstrap():
        return jsonify(service.bootstrap())

    @app.get('/api/health')
    def health():
        return jsonify(status='ready',models_loaded=len(service.models))

    @app.post('/api/predict')
    def predict():
        try:
            if not request.is_json:
                raise ValueError('Send a JSON object containing model, condition and features.')
            return jsonify(service.predict(request.get_json()))
        except BadRequest:
            return jsonify(error='Invalid JSON. Enter all 16 numeric feature values.'),400
        except ValueError as exc:
            return jsonify(error=str(exc)),400

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        return jsonify(error='Request exceeds the 32 KB demo limit.'),413

    @app.after_request
    def headers(response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    return app
