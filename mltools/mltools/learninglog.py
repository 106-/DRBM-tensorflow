# -*- coding: utf-8 -*-

import numpy as np
import json


class LearningLog:
    def __init__(self, learning_settings={}):
        self.testament = {}
        self.testament["setting"] = learning_settings
        self.testament["log"] = {}

    def make_log(self, learning_count, value_name, values):
        if not value_name in self.testament["log"]:
            self.testament["log"][value_name] = []
        if isinstance(values, list):
            self.testament["log"][value_name].append([learning_count] + values)
        else:
            self.testament["log"][value_name].append([learning_count, values])

    def save(self, filename):
        json.dump(self.testament, open(filename, "w+"), indent=2)
