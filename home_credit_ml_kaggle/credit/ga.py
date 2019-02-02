import gc
import json
import numpy as np
import pickle
import os
import pandas as pd
import random
import time
from threading import Thread
from credit.models import kfold_lightgbm
from credit.utils import (
    get_file, safe_load_json, dump_json,
    RAW_GENE, ART_GENE, PLUS,
    SUBTRACT, DIVIDE, MULTIPLY, folder_date,
    sample_ga_model, sample_params
)


SAMPLE = False
OFF = False
USE_GPU = False if OFF else True
MULTI = True


class TrainConfig:
    def __init__(self):
        self.version = 'v1'
        self.num_folds = 5
        self.stratified = True
        self.seed = 290
        self.use_gpu = USE_GPU
        self.gpu_devices = [0]
        self.ckpt_dir = None
        self.ga_model = {
            'pop_size': 20,
            'chromosome_size': 300,
            'workers': 2,
            'lazy_size': 30,  # lazy size per worker
            'mate_method': 0,
            'mate_numbers': 6,  # numbers of chromosomes to mate when moving to next gen
            'pop_reverse_sort': True,
            'mutate_chance': .5,
            'log_threshold': .7,
            'mutate_scale': 2,
        }
        self.params = {
            "boosting_type": "gbdt",
            "max_depth": 8,
            "seed": 0,
            "learning_rate": 0.02,
            "metric": "auc",
            "verbose": -1,
            "min_child_weight": 60,
            "colsample_bytree": 0.9497036,
            "nthread": 6,
            "num_leaves": 20,
            "reg_alpha": 0.041545473,
            "subsample_freq": 1,
            "objective": "binary",
            "min_split_gain": 0.0222415,
            "device_type": "cpu",
            "subsample": 0.8715623,
            "gpu_device_id": 1,
            "reg_lambda": 0.0735294,
            "xgboost_dart_mode": True,
        }
        if SAMPLE:
            self.ga_model = sample_ga_model
            self.params = sample_params

    def copy(self):
        return self.from_json(_json=self.to_json())

    @classmethod
    def from_json(cls, path=None, _json=None):
        obj = cls()
        assert path or _json, 'Either path to json file or json string must be passed'

        if path:
            with open(path) as fp:
                config = json.load(fp)
        else:
            config = json.loads(_json)

        obj.__dict__.update(config)
        return obj

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class LazyBin:

    def __init__(self, odf, max_size=5, version='v1', is_sample=False):
        self.bins = {}
        self.odf = odf
        self.file = 'bin_sample_{}.csv' if is_sample else 'bin_{}.csv'
        self.bin_path = os.path.join(get_file('genes_bin_dir', version), self.file)
        self.max_size = max_size
        self.version = version

    def __getitem__(self, item):

        if item == 'odf':
            """return original dataframe"""
            return self.odf

        if item not in self.bins:
            self.bins[item] = pd.read_csv(self.bin_path.format(item))

        self.update_cache(item)

        return self.bins[item]

    def update_cache(self, key):

        diff = len(self.bins) - self.max_size
        if diff > 0:
            can_delete = list(self.bins.keys())
            can_delete.pop(can_delete.index(key))
            [self.bins.pop(can_delete.pop()) for _ in range(diff)]


class Worker:

    def __init__(self, target, args, name, lazy_genes):
        self.thread = Thread(target=target, args=(*args, self))
        self.thread.name = name
        self.name = name
        self.thread.daemon = True
        self.finished = False
        self.train_config = None
        self.gpu_device = None
        self.lazy_genes = lazy_genes

    def start(self):
        print('Starting {}'.format(self))
        self.thread.start()

    def __str__(self):
        return 'Worker <{}>'.format(self.name)

    def __repr__(self):
        return self.__str__()


class LazyGenes:

    def __init__(self, max_size):
        self.genes = {}
        self.max_size = max_size

    def __getitem__(self, gene):
        if gene not in self.genes:
            self.genes[gene] = gene.get_table()
        self.update_cache(gene)
        return self.genes[gene]

    def update_cache(self, gene):

        diff = len(self.genes) - self.max_size
        if diff > 0:
            can_delete = list(self.genes.keys())
            can_delete.pop(can_delete.index(gene))
            [self.genes.pop(can_delete.pop()) for _ in range(diff)]
            gc.collect()


class Gene:

    def __init__(self, _id, df, _type=RAW_GENE, op=None, parents=None):
        self.id = _id
        self.type = _type
        self.op = op
        self.parents = parents
        self.df = df

    def apply_ops(self, other):
        return self + other, self - other, self * other, self / other

    def get_table(self):
        if self.type == RAW_GENE:
            return self.df[self.id]

        p1, p2 = self.parents
        tab = None
        if self.op == PLUS:
            tab = self.df[p1] + self.df[p2]
        elif self.op == SUBTRACT:
            tab = self.df[p1] - self.df[p2]
        elif self.op == DIVIDE:
            tab = self.df[p1] / self.df[p2]
        elif self.op == MULTIPLY:
            tab = self.df[p1] * self.df[p2]
        tab.name = self.id

        return tab

    def __add__(self, other):
        parents = [self.id, other.id]
        new_id = self.id + PLUS + other.id
        return Gene(new_id, self.df, ART_GENE, PLUS, parents=parents)

    def __sub__(self, other):
        parents = [self.id, other.id]
        new_id = self.id + SUBTRACT + other.id
        return Gene(new_id, self.df, ART_GENE, SUBTRACT, parents=parents)

    def __truediv__(self, other):
        parents = [self.id, other.id]
        new_id = self.id + DIVIDE + other.id
        return Gene(new_id, self.df, ART_GENE, DIVIDE, parents=parents)

    def __mul__(self, other):
        parents = [self.id, other.id]
        new_id = self.id + MULTIPLY + other.id
        return Gene(new_id, self.df, ART_GENE, MULTIPLY, parents=parents)

    def __str__(self):
        return 'Gene({})'.format(self.id)

    def __repr__(self):
        return self.__str__()


class Chromosome:

    def __init__(self, code, mate_method, genes, mutate_scale):
        self.code = code
        self.score = .0
        self.importance = None  # sort genes in code in accordance to importance
        self.size = len(self.code)
        self.mate_method = mate_method
        self.mutated = False
        self.model_path = None
        self.genes = genes
        self.mutate_scale = mutate_scale

    def set_importance(self, imp):
        imp = imp[['feature', 'importance']]
        imp = imp.groupby('feature').mean().sort_values(by='importance', ascending=False)
        self.importance = imp

    def mate(self, other):
        if self.mate_method == 0:
            imp = pd.concat([self.importance, other.importance])
            imp = imp.groupby('feature').mean().sort_values(by='importance', ascending=False).index.tolist()
            code = [g for g in self.genes if g.id in imp[:self.size]]
            return Chromosome(code, self.mate_method, self.genes, self.mutate_scale),

    def mutate(self):
        """mutate a random gene with a new gene not present in chromosome"""
        for _ in range(self.mutate_scale):
            ix = np.random.randint(0, self.size)
            rand_gene = random.choice([x for x in self.genes if x not in self.code])
            self.code[ix] = rand_gene
        self.mutated = True

    def is_trainable(self):
        if not self.score or self.mutated:
            return True
        return False

    def __add__(self, other):
        assert self.size == other.size, 'both chromosomes must have the same len'
        return self.mate(other)

    def __iter__(self):
        yield from self.code

    def __str__(self):
        if len(self.code) > 1:
            return 'Chromosome({:.3f}) <[{}, ..., {}]>'.format(self.score, self.code[0], self.code[-1])
        return 'Chromosome <{}>'.format(self.code)

    def __repr__(self):
        return self.__str__()


class Population:

    def __init__(self,  genes, tc):

        self.tc = tc
        self.genes = genes
        self.gam = self.tc.ga_model

        assert self.gam['mate_numbers'] % 2 == 0, 'Numbers of chromosomes to mate must be an even number'
        assert self.gam['pop_size'] >= self.gam['mate_numbers'], ('Numbers of chromosomes to mate cannot '
                                                                  'be greater than population size')

        self.gen = 0  # generation
        self.size = self.gam['pop_size']  # population size
        self.cs_size = self.gam['chromosome_size']  # chromosome size
        self.mate_method = self.gam['mate_method']
        self.mate_num = self.gam['mate_numbers']
        self.mutate_chance = self.gam['mutate_chance']
        self.sort_type = self.gam['pop_reverse_sort']

        self.members = self.gen_population()

    def gen_population(self):
        """ generate initial population"""
        pop = [Chromosome(random.sample(self.genes, self.cs_size),
                          self.mate_method, self.genes,
                          self.tc.ga_model['mutate_scale'])
               for _ in range(self.size-1)]
        return pop

    def sort(self):
        self.members = sorted(self.members, key=lambda x: x.score, reverse=self.sort_type)

    def mate_members(self):
        to_mate = self.members[:self.mate_num]
        next_gens = []
        for i in range(0, len(to_mate), 2):
            next_gens.append(*(self.members[i] + self.members[i+1]))

        next_gens = next_gens[:self.size]
        weak_nums = self.size - len(next_gens)
        self.members = self.members[:weak_nums] + next_gens

        assert len(self.members) == self.size, 'Error when mating members'

    def mutate_members(self):
        """Mutate by chance"""
        for mem in self.members:
            if np.random.rand() > self.mutate_chance:
                mem.mutate()

    def __str__(self):
        return 'Population <members:{} | genes:{}>'.format(self.size, self.cs_size)

    def __repr__(self):
        return self.__str__()


class CreditGA:

    def __init__(self, tc=TrainConfig(), sample_df=None):
        self.tc = tc
        self.gam = self.tc.ga_model

        ckpt_root = get_file('ga_ckpt', v=self.tc.version)
        os.makedirs(ckpt_root, exist_ok=True)
        self.tc.ckpt_dir = self.ckpt_dir = os.path.join(ckpt_root, folder_date())
        os.makedirs(self.ckpt_dir)
        print('Logging to {}'.format(self.ckpt_dir))
        self.ckpt_temp = os.path.join(self.ckpt_dir, '{gen}_{score}.pk')

        self.best_score = 0
        self.best_gen = 0
        self.workers = {}
        # self.lazy_genes = LazyGenes(self.gam['lazy_size'])

        self.odf = sample_df if sample_df is not None else pd.read_csv(get_file('all_data', self.tc.version))
        self.train_ix = self.odf[self.odf['TARGET'].notnull()].index
        self.test_ix = self.odf[self.odf['TARGET'].isnull()].index

        print('Generating genes')
        original_genes = self.odf.drop(['SK_ID_CURR', 'TARGET'], axis=1).columns.tolist()
        ogs = [Gene(i, df=self.odf) for i in original_genes]
        self.genes = ogs
        new_genes = []
        for ix, g in enumerate(self.genes[:-1]):
            for og in self.genes[ix+1:]:
                new_genes.extend(g.apply_ops(og))
        self.genes.extend(new_genes)
        print('Generated {} genes!'.format(len(self.genes)))

        # Generate initial generation
        self.population = Population(genes=self.genes, tc=self.tc)
        org_chromo = ogs[:self.gam['chromosome_size']]
        diff = self.gam['chromosome_size'] - len(org_chromo)
        random.shuffle(new_genes)
        org_chromo += new_genes[:diff]
        assert len(org_chromo) == self.gam['chromosome_size'], 'Error in original chromo size'
        self.population.members.append(Chromosome(org_chromo,  self.gam['mate_method'],
                                                  self.genes,  self.gam['mutate_scale']))
        assert len(self.population.members) == self.population.size, 'Discrepancies in population size'
        print('Generated a population of size {}'.format(self.population.size))

        if self.tc.use_gpu:
            alloc = self.gam['workers'] // len(self.tc.gpu_devices)
            rem = self.gam['workers'] % len(self.tc.gpu_devices)
            self.gpu_alloc = self.tc.gpu_devices * (alloc + rem)

    def get_chrome_df(self, chrome, worker):
        df = pd.DataFrame()
        for gene in chrome:
            df[gene.id] = worker.lazy_genes[gene]
        df[['SK_ID_CURR', 'TARGET']] = self.odf[['SK_ID_CURR', 'TARGET']]
        return df.loc[self.train_ix], df.loc[self.test_ix]

    def get_trainables(self):
        return [mem for mem in self.population.members if mem.is_trainable()]

    def train_population(self, members, worker):
        for chrome in members:
            if chrome.is_trainable():
                train_df, test_df = self.get_chrome_df(chrome, worker)
                imp, chrome.score, chrome.model_path = kfold_lightgbm(
                    worker.train_config, manual=True,
                    train_df=train_df, test_df=test_df
                )
                chrome.set_importance(imp)
                chrome.mutated = False

        worker.finished = True
        print('{} is done!'.format(worker.name))

    def assign_workers(self):
        """split population members and assign to workers for training"""
        workers = {}
        trainables = self.get_trainables()
        dividend = len(trainables) // self.gam['workers']
        dividend += (len(trainables) % self.gam['workers'] > 0)

        for d in range(self.gam['workers']):
            start_ix = d * dividend
            stop_ix = start_ix + dividend
            mem = trainables[start_ix: stop_ix]
            workers[d] = Worker(
                target=self.train_population, args=(mem, ),
                name='Worker_{}'.format(d), lazy_genes=LazyGenes(self.gam['lazy_size'])
            )
            workers[d].train_config = self.tc.copy()

            if self.tc.use_gpu:
                workers[d].train_config.params['device_type'] = 'gpu'
                workers[d].train_config.params['gpu_device_id'] = self.gpu_alloc[d]

        self.workers = workers

    def await_workers(self):
        """check to make sure all workers have finished their tasks"""
        start_time = time.time()
        while True:
            _bool = True
            for w in self.workers.values():
                _bool &= w.finished
                print(w, ' : ', w.finished)

            if _bool:
                break

            time.sleep(600)

        print('All workers have finished in {}.'.format(time.time() - start_time))

    def start_workers(self):
        print('Starting {} workers'.format(len(self.workers)))
        for w in self.workers.values():
            w.start()

    def evolve(self):
        e = 1
        try:
            while e:
                self.assign_workers()
                self.start_workers()
                self.await_workers()
                self.population.sort()

                cs = self.population.members[0].score
                gen = self.population.gen
                if cs >= self.best_score:
                    self.best_score = cs
                    self.best_gen = gen
                rank_file = 'sample_ga_rank' if SAMPLE else 'ga_rank'
                rank_path = get_file(rank_file, self.tc.version)
                current_rank = safe_load_json(rank_path)
                current_rank[self.population.members[0].model_path] = {
                    'gen': self.population.gen,
                    'score': cs
                }
                dump_json(current_rank, rank_path)
                with open(self.ckpt_temp.format(gen=gen, score=cs), 'wb') as fp:
                    pickle.dump(self.population.members, fp)

                print('\n-----------------------------------------------------------------------------')
                print('Gen: {} | Best score: {} by {} | Current Best Score: {}'.format(
                    gen, self.best_score, self.best_gen, cs))
                print('-----------------------------------------------------------------------------\n')

                self.population.mate_members()
                self.population.mutate_members()
                self.population.gen += 1
                gc.collect()

                if SAMPLE:
                    e -= 1
        except KeyboardInterrupt:
            print('Exited Gracefully with score:{}'.format(self.best_score))

        print('Finished!')
