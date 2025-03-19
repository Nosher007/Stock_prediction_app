from flask import Blueprint, jsonify

api = Blueprint('api', __name__)
@api.route('/hellos',methods=['GET'])
def hello():
    return jsonify({'message':'Hello World'})