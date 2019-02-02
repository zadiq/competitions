import time
from threading import Thread
from multiprocessing import Process, Queue, Event


class Worker:

    def __init__(self, target, task, name, func_args=None, use_process=False):
        func_args = func_args or []
        self.queue = Queue()
        self.finished_event = Event()
        if use_process:
            self.executor = Process(target=target, args=(task, self, *func_args))
        else:
            self.executor = Thread(target=target, args=(task, self, *func_args))
        self.executor.name = name
        self.name = name
        self.executor.daemon = True
        self.finished = False
        self.train_config = None
        self.error_occurred = False
        self.error = Exception

    def start(self):
        print('Starting {}'.format(self))
        self.executor.start()

    def __str__(self):
        return 'Worker <{}>'.format(self.name)

    def __repr__(self):
        return self.__str__()


class GroupWorkers:

    def __init__(self, num, func, tasks, func_args=None, name=None, use_process=False):
        self.name = name
        self.num_of_workers = num
        self.func = func
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.workers = {}
        # TODO change assignment algo to loop through task
        dividend = self.num_tasks // self.num_of_workers
        dividend += (self.num_tasks % self.num_of_workers > 0)
        self.use_process = use_process
        for d in range(self.num_of_workers):
            start_ix = d * dividend
            stop_ix = start_ix + dividend
            task = self.tasks[start_ix: stop_ix]
            self.workers[d] = Worker(
                target=self.func, task=task,
                name='Worker_{}'.format(d), func_args=func_args,
                use_process=use_process
            )

    def start(self):
        for w in self.workers.values():
            w.start()

    def await(self, func=None, t=5):
        while True:
            if func:
                func()

            if self.use_process and self.process_is_finished:
                break

            if self.is_finished:
                break

            time.sleep(t)

    @property
    def process_is_finished(self):
        _bool = True
        for w in self.workers.values():
            print(w, ': ', w.finished_event.is_set())
            _bool &= w.finished_event.is_set()

        return _bool

    @property
    def is_finished(self):
        _bool = True
        for w in self.workers.values():
            _bool &= w.finished
            if w.error_occurred:
                raise w.error

        return _bool

    @property
    def status(self):
        return dict(zip(list(self.workers.values()), list(map(lambda w: w.finished, self.workers.values()))))

    def __str__(self):
        return 'GroupWorker <{}:{}>'.format(self.name, self.num_of_workers)

    def __repr__(self):
        return self.__str__()
