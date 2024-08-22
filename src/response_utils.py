from flask import jsonify


class ResponseUtils:
    @staticmethod
    def send_response(status_code, data):
        response = jsonify(data)
        response.status_code = status_code
        return response
