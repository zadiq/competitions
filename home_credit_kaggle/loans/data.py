from loans import utils
import json
import csv
import ast
import numpy as np
import random


random.seed(21)
np.random.seed(21)


def get_data_gen(mode, batch_size, debug=False):
    options = [utils.TRAIN_MODE, utils.EVAL_MODE, utils.TEST_MODE]
    assert mode in options, 'Invalid mode provided {}. Select from {}'.format(mode, options)

    if mode == utils.TRAIN_MODE:
        app_file = utils.get_file('re_app_train')
        meta_ix = 'train_size'
    elif mode == utils.EVAL_MODE:
        app_file = utils.get_file('re_app_eval')
        meta_ix = 'eval_size'
    else:
        app_file = utils.get_file('re_app_test')
        meta_ix = 'test_size'

    with open(utils.get_file('app_unique_voc')) as apv:
        vocab = json.load(apv)
    with open(utils.get_file('re_meta')) as meta_fp:
        meta_data = json.load(meta_fp)

    def parse(_data, _ordered_column):
        _ = _data.pop('SK_ID_CURR')
        target = int(_data.pop('TARGET'))

        for d in _data:
            if d in vocab:
                # replace categorical columns with their index in vocabulary list
                _data[d] = vocab[d].index(_data[d])
            else:
                _data[d] = ast.literal_eval(_data[d])

        assert len(_data) == len(_ordered_column), 'Error when parsing data'
        ordered_data = [_data[d] for d in _ordered_column]

        return ordered_data, [target]

    try:
        with open(app_file) as app_fp:
            app_reader = csv.reader(app_fp)
            columns = next(app_reader)
            ordered_column = columns.copy()
            ordered_column.pop(ordered_column.index('TARGET'))
            ordered_column.pop(ordered_column.index('SK_ID_CURR'))

            epoch = 1
            batch_ix = 0
            data_ix = 0
            steps = 0

            while True:
                batch_features = []
                batch_target = []
                try:
                    for i in range(batch_size):
                        data = next(app_reader)
                        data_ix += 1
                        f, t = parse(dict(zip(columns, data)), ordered_column)
                        batch_features.append(f)
                        batch_target.append(t)
                except StopIteration:
                    app_fp.seek(0)
                    app_reader = csv.reader(app_fp)
                    _ = next(app_reader)  # skip column keys
                finally:
                    if debug:
                        steps += 1
                        batch_ix += 1
                        data_size = len(batch_features)

                        params = {
                            'epoch': epoch,
                            'batch': batch_ix,
                            'data_size': data_size,
                            'steps': steps,
                            'data_ix': data_ix
                        }
                        print(('epoch: {epoch} | steps: {steps} | batch: {batch} |'
                               ' data_size: {data_size} | data_ix: {data_ix}').format(**params))

                        # reset values for new epoch
                        if data_ix >= meta_data[meta_ix]:
                            data_ix = 0
                            batch_ix = 0
                            epoch += 1
                            print('Starting new {} epoch: {}'.format(mode, epoch))

                    yield utils.np.array(batch_features), utils.np.array(batch_target)

    except KeyboardInterrupt:
        print('Exiting {} generator gracefully!'.format(mode))


class LoanData:

    def __init__(self, mode, batch_size=100, merge=True, verbose=False):

        options = [utils.TRAIN_MODE, utils.EVAL_MODE, utils.TEST_MODE]
        assert mode in options, 'Invalid mode provided {}. Select from {}'.format(mode, options)

        self.mode = mode
        self.batch_size = batch_size
        self.merge = merge
        self.verbose = verbose
        self.data_point = 0

        with open(utils.get_file('merged_meta')) as mm:
            self.meta = json.load(mm)

        cols = {
            'PREV': [],
            'CC': [],
            'BURO': [],
            'INSTAL': [],
            'POS': []
        }
        for c in self.meta['input_order'].values():
            if c.startswith('PREV_'):
                cols['PREV'].append(c)
            elif c.startswith('CC_'):
                cols['CC'].append(c)
            elif c.startswith('BURO_'):
                cols['BURO'].append(c)
            elif c.startswith('INSTAL_'):
                cols['INSTAL'].append(c)
            elif c.startswith('POS_'):
                cols['POS'].append(c)

        self.input_spec = {
            'APP_DATA': (len(self.meta['app_cols']) - 2, ),  # - SK_ID_CURR and TARGET
            'APP_NOT_AVL_DATA': (len(self.meta['not_avl_col']), ),
            'PREV_APP_DATA': (len(cols['PREV']), ),
            'CC_DATA': (len(cols['CC']), ),
            'BURO_DATA': (len(cols['BURO']), ),
            'INSTAL_DATA': (len(cols['INSTAL']), ),
            'POS_DATA': (len(cols['POS']), ),
            'MERGED': (928, ),
        }
        if self.merge:
            self.inputs = ['MERGED']
        else:
            self.inputs = ['APP_DATA', 'APP_NOT_AVL_DATA', 'PREV_APP_DATA',
                           'POS_DATA', 'CC_DATA', 'BURO_DATA', 'INSTAL_DATA']

        if mode == utils.TRAIN_MODE:
            self.data_file = utils.get_file('model_train')
            print('{} training data | {} TARGET 1 | {} TARGET 0'.format(
                self.meta['model_train_size'], self.meta['model_train_class_one_size'],
                self.meta['model_train_class_zero_size']
            ))
        elif mode == utils.EVAL_MODE:
            self.data_file = utils.get_file('model_eval')
            print('{} eval data | {} TARGET 1 | {} TARGET 0'.format(
                self.meta['model_eval_size'], self.meta['model_eval_class_one_size'],
                self.meta['model_eval_class_zero_size']
            ))
        else:
            self.data_file = utils.get_file('model_test')
            print('{} test data | {} TARGET 1 | {} TARGET 0'.format(
                self.meta['model_test_size'], self.meta['model_test_class_one_size'],
                self.meta['model_test_class_zero_size']
            ))

        self.l_ = ast.literal_eval
        print('Getting {} data from {}.'.format(mode, self.data_file))
        self.data_fp = open(self.data_file)
        self.data_reader = csv.reader(self.data_fp)
        self.columns = next(self.data_reader)

    def parse(self, _data):
        _data = dict(zip(self.columns, _data))
        app_id = _data.pop('SK_ID_CURR')
        _data.pop('APP_index')
        target = self.l_(_data.pop('APP_TARGET'))

        app_data = []
        app_not_avl = []
        prev_data = []
        pos_data = []
        cc_data = []
        ins_data = []
        buro_data = []
        others = []

        for d, v in _data.items():
            try:
                if d in self.meta['app_cols']:
                    app_data.append(self.l_(v))
                elif d in self.meta['not_avl_col']:
                    app_not_avl.append(self.l_(v))
                elif d.startswith('PREV_'):
                    prev_data.append(self.l_(v))
                elif d.startswith('CC_'):
                    cc_data.append(self.l_(v))
                elif d.startswith('POS_'):
                    pos_data.append(self.l_(v))
                elif d.startswith('INSTAL_'):
                    ins_data.append(self.l_(v))
                elif d.startswith('BURO_'):
                    buro_data.append(self.l_(v))
                else:
                    print('Doesnt fit a cat', d)
                    others.append(self.l_(v))
            except ValueError:
                print('Error in data', app_id, d, v)
                raise

        assert (len(app_data) + len(app_not_avl) + len(prev_data) +
                len(pos_data) + len(cc_data) + len(ins_data) +
                len(buro_data) == len(_data)), 'Problem in data integrity'

        if self.merge:
            _data = app_data + app_not_avl + prev_data + pos_data + cc_data + buro_data + ins_data
            _data = [np.array(_data)]
        else:
            _data = [np.array(app_data), np.array(app_not_avl),
                     np.array(prev_data), np.array(pos_data),
                     np.array(cc_data), np.array(buro_data),
                     np.array(ins_data)]

        return _data, [target]

    def data_gen(self):
        while True:
            batch_input = [[] for _ in self.inputs]
            batch_target = []
            try:
                for _ in range(self.batch_size):
                    data = next(self.data_reader)
                    self.data_point += 1
                    inputs, target = self.parse(data)
                    [batch_input[n].append(inputs[n]) for n in range(len(batch_input))]
                    batch_target.append(target)
            except StopIteration:
                if self.verbose:
                    print('Completed an epoch')
                self.data_point = 0
                self.data_fp.seek(0)
                self.data_reader = csv.reader(self.data_fp)
                _ = next(self.data_reader)  # skip columns
            finally:
                if self.verbose:
                    print('At sample {}'.format(self.data_point))
                batch_input = [np.array(i) for i in batch_input]
                yield batch_input, np.array(batch_target)


# if __name__ == '__main__':
#     train_data = LoanData('eval_mode', 40, merge=True, verbose=True)
#     gen = train_data.data_gen()
