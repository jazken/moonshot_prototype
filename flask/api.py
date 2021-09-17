from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from flask_cors import CORS, cross_origin

# ML model dependencies
import torch
from transformers.file_utils import (
    is_tf_available,
    is_torch_available,
    is_torch_tpu_available,
)
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
import random
from sklearn.model_selection import train_test_split

app = Flask(__name__)
api = Api(app)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

TODOS = {
    "todo1": {"task": "build an API"},
    "todo2": {"task": "?????"},
    "todo3": {"task": "profit!"},
}


def abort_if_todo_doesnt_exist(todo_id):
    if todo_id not in TODOS:
        abort(404, message="Todo {} doesn't exist".format(todo_id))


parser = reqparse.RequestParser()
parser.add_argument("task")
parser.add_argument("sentence")


# Todo
# shows a single todo item and lets you delete a todo item
class Todo(Resource):
    def get(self, todo_id):
        abort_if_todo_doesnt_exist(todo_id)
        return TODOS[todo_id]

    def delete(self, todo_id):
        abort_if_todo_doesnt_exist(todo_id)
        del TODOS[todo_id]
        return "", 204

    def put(self, todo_id):
        args = parser.parse_args()
        task = {"task": args["task"]}
        TODOS[todo_id] = task
        return task, 201


# TodoList
# shows a list of all todos, and lets you POST to add new tasks
class TodoList(Resource):
    def get(self):
        return TODOS

    def post(self):
        args = parser.parse_args()
        todo_id = int(max(TODOS.keys()).lstrip("todo")) + 1
        todo_id = "todo%i" % todo_id
        TODOS[todo_id] = {"task": args["task"]}
        return TODOS[todo_id], 201


def load_model(model_file):
    """To load the model before the app begins"""
    model_name = "bert-base-uncased"
    model_state_dict = torch.load(model_file)
    loaded_model = BertForSequenceClassification.from_pretrained(
        model_name, state_dict=model_state_dict, num_labels=2, output_attentions=True
    )
    loaded_model.eval()
    return loaded_model


def inference(sentence):
    # load model first
    loaded_model = load_model("./results/checkpoint-1500/pytorch_model.bin")
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(
        model_name, do_lower_case=True)
    inputs = tokenizer.encode(sentence, return_tensors="pt")
    labels = ["not_grooming", "grooming"]
    outputs = loaded_model(inputs)
    probs = outputs[0].softmax(1)
    return labels[probs.argmax()]


# Grooming api
# lets you POST to get an inference result

class Grooming(Resource):
    def get(self):
        return NotImplementedError

    @cross_origin()
    def post(self):
        args = parser.parse_args()
        text_input = args["sentence"]
        label = inference(text_input)
        if label == "grooming":
            return "WARNING - ONLINE GROOMING DETECTED", 201
        else:
            return "NO ISSUE", 201


##
# Actually setup the Api resource routing here
##
api.add_resource(TodoList, "/todos")
api.add_resource(Grooming, "/grooming")
api.add_resource(Todo, "/todos/<todo_id>")


if __name__ == "__main__":

    app.run(debug=True)
