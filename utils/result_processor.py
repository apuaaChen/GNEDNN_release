import json
import os
import sys


class Results:
    """
    Record the results and store them to the json files
    """
    def __init__(self, list_name, exp_name, json_dir):
        self.exp_name = exp_name
        if not os.path.exists(json_dir):
            os.mkdir(json_dir)

        self.exp_path = json_dir + exp_name + '.json'
        self.list_path = json_dir + list_name + '.json'

        # See whether this experiment exists
        if os.path.exists(self.exp_path):
            print('This experiment has ben run.')
            sys.exit()

        # create the dictionary container for results to be recorded

        self.result = {
            'name': exp_name,
            'epoch': [],
            'test_acc': [],
            'train_acc': [],
            'train_loss': [],
        }

    def write_result(self):
        # log the experiment into the list json file
        if os.path.exists(self.list_path):
            with open(self.list_path) as json_file:
                exp_list = json.load(json_file)
        else:
            exp_list = {
                'exps': []
            }

        exp_list['exps'].append(self.exp_name + '.json')

        with open(self.list_path, 'w') as json_file:
            json.dump(exp_list, json_file)

        # write json file
        with open(self.exp_path, 'w') as file:
            json.dump(self.result, file)
