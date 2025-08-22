import torch
from einops import rearrange
import gc

class AttentionControl_v2:
    def __init__(self):
        self.batch_id = 0
        self.step_store = self.get_empty_store()
        self.attention_store_batch = []
        self.attention_store_step = []
        self.cur_step = 0
        self.total_step = 0
        self.cur_index = 0
        self.cur_indexKey = 0
        self.init_store = False
        self.restore = False

    @staticmethod
    def get_empty_store():
        return {
            'query': [],
            'key': [],
            'value': [],
            'first_key': [],
            'first_value': []
        }

    def set_step(self, step):
        self.cur_step = step
        if step == 0 and self.batch_id != 0:
            self.attention_store_step.append(self.step_store)
            self.attention_store_batch = self.attention_store_step
            self.attention_store_step = []
        elif step != 0:
            self.attention_store_step.append(self.step_store)

        self.clear_store()
        self.cur_index = 0
        self.cur_indexKey = 0

    def set_total_step(self, total_step):
        self.total_step = total_step
        self.cur_index = 0
        self.cur_indexKey = 0

    def clear_store(self):
        del self.step_store
        torch.cuda.empty_cache()
        gc.collect()
        self.step_store = self.get_empty_store()

    def set_task(self, task):
        self.init_store = False
        self.restore = False
        self.cur_index = 0    # recode query index
        self.cur_indexKey = 0  # recode key index

        if 'init_first' in task:
            self.attention_store_batch = []
            self.attention_store_step = []
            self.init_store = True
            self.clear_store()

        if 'next_batch' in task:
            self.restore = True
            self.attention_store_batch = []

    def store_query(self, query):
        # 对于quer，1、对每一个batch最后一帧的query进行保存，2、并判断是否需要get上一batch的query
        self.step_store['query'].append(query.detach().clone().cpu())   
        if self.init_store and self.batch_id == 0:
            return None
        elif self.restore and self.batch_id != 0:
            return self.attention_store_batch[self.cur_step]['query'][self.cur_index]
        else:
            assert False, 'store_query error'


############### 
    def store_key(self, key):
        self.step_store['key'].append(key.detach().clone().cpu())
        if not (self.init_store and self.batch_id == 0) and not (self.restore and self.batch_id != 0):
            assert False, 'store_key error'

    def get_batch_pre_key(self):
        if self.restore and self.batch_id != 0:
            return self.attention_store_batch[self.cur_step]['key'][self.cur_indexKey]

    def store_value(self, value):
        self.step_store['value'].append(value.detach().clone().cpu())
        if not (self.init_store and self.batch_id == 0) and not (self.restore and self.batch_id != 0):
            assert False, 'store_value error'

    def get_batch_pre_value(self):
        if self.restore and self.batch_id != 0:
            return self.attention_store_batch[self.cur_step]['value'][self.cur_indexKey]


##################
    def store_first_key(self, first_key):
        self.step_store['first_key'].append(first_key.detach().clone().cpu())
        if not (self.init_store and self.batch_id == 0):
            assert False, 'store_first_key error'

    def get_first_key(self):
        if self.restore and self.batch_id != 0:
            first_key = self.attention_store_batch[self.cur_step]['first_key'][self.cur_indexKey]
            self.step_store['first_key'].append(first_key)
            return first_key

    def store_first_value(self, first_value):
        self.step_store['first_value'].append(first_value.detach().clone().cpu())
        if not (self.init_store and self.batch_id == 0):
            assert False, 'store_first_value error'

    def get_first_value(self):
        if self.restore and self.batch_id != 0:
            first_value = self.attention_store_batch[self.cur_step]['first_value'][self.cur_indexKey]
            self.step_store['first_value'].append(first_value)
            return first_value