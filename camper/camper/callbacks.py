

def trace_callback(self):

    def callback(res):
        _ = res
        self.opt_trace.append(self.opt_clf.best_score_)
        log = {self.name: self.opt_trace}
        self.logger(log, _type='trace')
        if self.ckpt_score_is_better():
            self.logger('{}: saving at ckpt:{}, score:{}'.format(self.name, self.ckpt_counter,
                                                                 self.opt_clf.best_score_))
            self.logger(self, _type='model', file_name=self.filename)
        else:
            self.logger('{}: NOT saving at ckpt:{}, score:{}'.format(self.name, self.ckpt_counter,
                                                                     self.opt_clf.best_score_))
        self.opt_ckpt_best_score = self.opt_clf.best_score_
        self.ckpt_counter += 1

    return callback
