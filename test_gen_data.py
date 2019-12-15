import unittest
import pandas as pd

from gen_data import get_empty_val_dict
from gen_data import fill_val_dict
from gen_data import get_avg_val_dict
from gen_data import get_sd_val_dict
from gen_data import gen_new_data

class TestDicts(unittest.TestCase):
    def setUp(self):
        data = {'2019-12-01 13:30': [1,2,3,0], '2019-12-01 14:00': [0,0,0,-1]}
        self.test_data = pd.DataFrame.from_dict(data, orient='index')

    def test_get_empty_val_dict(self):
        self.assertEqual(get_empty_val_dict(self.test_data), {'13:30': [], '14:00': []})

    def test_fill_val_dict(self):
        self.test_data['Total - MW'] = self.test_data.sum(axis=1)
        val_dict = {'13:30': [], '14:00': []}
        self.assertEqual(fill_val_dict(self.test_data, val_dict), {'13:30': [6], '14:00': [-1]})

    def test_get_avg_val_dict(self):
        val_dict = {'13:30': [6,4], '14:00': [-1]}
        self.assertEqual(get_avg_val_dict(val_dict), {'13:30': 5, '14:00': -1})

    def test_get_sd_val_dict(self):
        val_dict = {'13:30': [1,1,1,1], '14:00': [100,0,100,0]}
        self.assertEqual(get_sd_val_dict(val_dict), {'13:30': 0, '14:00': 57.735026918962575})

class TestGenData(unittest.TestCase):
    def setUp(self):
        self.val_dict = {'13:30': [1,1,1,1], '14:00': [100,0,100,0]}
        self.avg_val_dict = {'13:30': 1, '14:00': 100}
        self.sd_val_dict = {'13:30': 0, '14:00': 57.735026918962575}

    def test_gen_new_data_outage(self):
        data_type = 'outage'
        days_to_gen = 10
        gen_data = gen_new_data(data_type, self.val_dict, self.avg_val_dict, self.sd_val_dict, days_to_gen)
        self.assertTrue(0 in gen_data.loc[0].tolist()[:-1])

    def test_gen_new_data_clean(self):
        data_type = 'clean'
        days_to_gen = 10
        gen_data = gen_new_data(data_type, self.val_dict, self.avg_val_dict, self.sd_val_dict, days_to_gen)
        self.assertFalse(0 in gen_data.loc[0].tolist()[:-1])
