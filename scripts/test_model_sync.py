"""Offline regression tests for the app-owned curator; no provider credentials."""
import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sync', ROOT / 'scripts/sync-model-catalog.py')
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


class CuratorTests(unittest.TestCase):
    def setUp(self):
        self.policy = {'providers': {'test': {'source':'example','default':'approved',
            'models':['approved'], 'thinking_modes':['effort_levels'],
            'thinking_parameters':['reasoning_effort'], 'required_level':'low',
            'cost_ceiling':{'input_cost_per_mtok':1}}}}
        self.catalog = {'version':1,'providers':{'example':[
            {'model_id':'approved','display_name':'Approved','model_type':'text',
             'input_cost_per_mtok':0.5,'output_cost_per_mtok':1,
             'thinking':{'supported':True,'mode':'effort_levels','parameter':'reasoning_effort','levels':['low','high']}}
        ]}}

    def curate(self):
        return sync.curate(json.dumps(self.catalog).encode(), self.policy)

    def test_deterministic_and_does_not_auto_add_unreviewed_models(self):
        new = copy.deepcopy(self.catalog['providers']['example'][0]); new['model_id']='unknown'
        self.catalog['providers']['example'].append(new)
        self.assertEqual(self.curate(), self.curate())
        self.assertEqual([m['id'] for m in self.curate()['providers']['test']['models']], ['approved'])

    def test_missing_default_and_deprecation_fail(self):
        self.catalog['providers']['example'][0]['deprecated'] = True
        with self.assertRaises(ValueError): self.curate()
        self.catalog['providers']['example'] = []
        with self.assertRaises(ValueError): self.curate()

    def test_price_cap_and_calling_changes_fail(self):
        m = self.catalog['providers']['example'][0]
        m['input_cost_per_mtok'] = 2
        with self.assertRaises(ValueError): self.curate()
        m['input_cost_per_mtok'] = 0.5
        m['thinking']['parameter'] = 'new.unsupported'
        with self.assertRaises(ValueError): self.curate()
        m['thinking']['parameter'] = 'reasoning_effort'
        m['thinking']['levels'] = ['high']
        with self.assertRaises(ValueError): self.curate()


if __name__ == '__main__': unittest.main()
