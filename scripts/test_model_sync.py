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

    def automatic(self):
        rule = self.policy['providers']['test']
        rule.pop('models')
        rule.update(selection='matching_catalog', include_patterns=[r'approved(?:-v\d+)?'])

    def test_matching_catalog_adds_new_compatible_releases(self):
        self.automatic()
        new = copy.deepcopy(self.catalog['providers']['example'][0])
        new['model_id'] = 'approved-v2'
        self.catalog['providers']['example'].insert(0, new)
        selected = self.curate()['providers']['test']
        self.assertEqual([m['id'] for m in selected['models']], ['approved-v2', 'approved'])
        self.assertEqual(selected['default'], 'approved')
        self.assertEqual(self.curate(), self.curate())

    def test_matching_catalog_records_specialized_and_retired_exclusions(self):
        self.automatic()
        for model_id, extra in [('approved-search', {}), ('approved-v2', {'deprecated': True}),
                                ('approved-v3', {'model_type': 'embedding'})]:
            new = copy.deepcopy(self.catalog['providers']['example'][0])
            new.update(model_id=model_id, **extra)
            self.catalog['providers']['example'].append(new)
        selected = self.curate()['providers']['test']
        self.assertEqual([m['id'] for m in selected['models']], ['approved'])
        self.assertEqual(len(selected['excluded_models']), 3)
        self.assertTrue(all(m['reason'] for m in selected['excluded_models']))

    def test_matching_catalog_rejects_incompatible_new_release(self):
        self.automatic()
        new = copy.deepcopy(self.catalog['providers']['example'][0])
        new['model_id'] = 'approved-v2'
        new['thinking']['parameter'] = 'unsupported_protocol'
        self.catalog['providers']['example'].append(new)
        with self.assertRaisesRegex(ValueError, 'unsupported thinking parameter'):
            self.curate()

    def test_matching_catalog_still_requires_a_live_default(self):
        self.automatic()
        self.catalog['providers']['example'][0]['deprecated'] = True
        with self.assertRaisesRegex(ValueError, 'default must be approved'):
            self.curate()


if __name__ == '__main__': unittest.main()
