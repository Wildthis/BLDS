from flask import jsonify

def make_response(data=None, status=True, message="Success"):
    response = {
        "code": 200,
        "status": status,
        "message": message,
        "data": data
    }
    return jsonify(response), status