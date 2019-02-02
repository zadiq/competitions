import os
import json
# import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.externals import joblib as pickle


def safe_json_load(path):
    try:
        with open(path) as fp:
            data = json.load(fp)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    return data


def json_dump(data, path):
    with open(path, 'w') as fp:
        json.dump(data, fp, indent=4)


def divider(end=False):
    if end:
        print('------------------------------------------------------------------\n')
        return
    print('\n------------------------------------------------------------------')


class StatusLogCheckpoint:

    def __init__(self, folder, meta=None,  read_only=False):
        self.folder = folder
        assert os.path.exists(folder), '{} does not exists'.format(folder)
        self.meta = meta or {
            'Name': '',
            'Description': ''
        }
        self.read_only = read_only
        self.file = os.path.join(folder, 'status.txt')
        self.imp_file = os.path.join(folder, 'importance.json')
        self.opt_file = os.path.join(folder, 'optimized.json')
        self.trace_file = os.path.join(folder, 'trace.json')
        self.trace_img = os.path.join(folder, 'trace.png')
        self.voter_file = os.path.join(folder, 'voter.json')
        self.models_file = os.path.join(folder, 'models.json')
        self.v_model_file = os.path.join(folder, 'voter.pkl')
        self.c_model_folder = os.path.join(folder, 'models')
        self.meta_file = os.path.join(folder, 'meta.json')
        os.makedirs(self.c_model_folder, exist_ok=True)

        if not read_only:
            self.log('Initiated Log Entries {}'.format(datetime.utcnow()))
            json_dump(self.meta, self.meta_file)

    def log(self, entry):
        if not entry.endswith('\n'):
            entry += '\n'
        with open(self.file, 'a+') as fp:
            fp.write(entry)

    @staticmethod
    def json_log(entry, path):
        cur_entry = safe_json_load(path)
        cur_entry.update(entry)
        json_dump(cur_entry, path)

    @staticmethod
    def json_trace_log(entry, path, key):
        cur_entry = safe_json_load(path)
        cur_entry[key] = cur_entry.get(key, [])
        cur_entry[key].append(entry)
        json_dump(cur_entry, path)

    @staticmethod
    def model_log(clf, path):
        # with open(path, 'wb') as fp:
        #     pickle.dump(clf, fp)
        pickle.dump(clf, path)

    @staticmethod
    def display_dict(obj):
        for key, val in obj.items():
            print('{}:'.format(key))
            for v in val:
                print('\t{}: {}'.format(v, val[v]))

    def display(self, _type='default'):
        if _type in ['default', 'mcfi', 'mco', 'vc']:
            divider()
            if _type == 'default':
                print('Status')
                with open(self.file,) as fp:
                    [print(l) for l in fp.readlines()]
            elif _type == 'mcfi':
                print('Feature Importance')
                self.display_dict(safe_json_load(self.imp_file))
            elif _type == 'mco':
                print('Optimized Classifiers')
                self.display_dict(safe_json_load(self.opt_file))
            elif _type == 'vc':
                print('Optimized Voter Classifiers')
                self.display_dict(safe_json_load(self.voter_file))
            divider(end=True)
        elif _type == 'trace':
            for clf, trace in safe_json_load(self.trace_file).items():
                plt.plot(trace, label=clf)
                plt.ylabel('Percentages')
                plt.xlabel('Iterations')
                plt.legend(loc=4)
            fig = plt.gcf()
            fig.savefig(self.trace_img)
            plt.show()

    def display_all(self, exclude=None):
        _types = ['default', 'mcfi', 'mco', 'trace', 'vc']
        if exclude:
            _types.pop(_types.index(exclude))

        for t in _types:
            self.display(t)

    def __call__(self, entry, _type='default', **kwargs):
        if not self.read_only:
            if _type == 'default':
                self.log(entry)
            elif _type == 'mcfi':
                self.json_log(entry, self.imp_file)
            elif _type == 'mco':
                self.json_log(entry, self.opt_file)
            elif _type == 'meta':
                self.json_log(entry, self.meta_file)
            elif _type == 'trace':
                self.json_log(entry, self.trace_file)
            elif _type == 'vc':
                self.json_log(entry, self.voter_file)
            elif _type == 'score_trace':
                self.json_trace_log(entry, self.trace_file, kwargs['id'])
            elif _type == 'model':
                path = os.path.join(self.c_model_folder, kwargs['file_name'])
                self.model_log(entry, path)
                log = {str(kwargs['id']) + '__' + kwargs['name']: path}
                self.json_log(log, self.models_file)
            elif _type == 'voter':
                self.model_log(entry, self.v_model_file)

    def __str__(self):
        return "Logger({})".format(self.folder)

    def __repr__(self):
        return self.__str__()
