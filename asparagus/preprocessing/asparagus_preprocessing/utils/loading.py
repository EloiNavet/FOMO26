import json


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)
