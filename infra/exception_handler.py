from flask import Flask, jsonify

from infra.make_response import make_response

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