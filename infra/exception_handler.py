from flask import Flask, jsonify
from functools import wraps
from infra.make_response import make_response
import logging

logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.errorhandler(404)
def handle_not_found_error(error):
    return make_response(status=False, message="Not Found")

@app.errorhandler(500)
def handle_internal_server_error(error):
    return make_response(status=False, message="Internal Server Error")

@app.errorhandler(Exception)
def handle_generic_error(error):
    return make_response(status=False, message=str(error))

def handle_exception(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}", exc_info=True)
            return make_response(code=500, message=str(e))
    return decorated_function