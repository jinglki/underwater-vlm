"""CPU-only regression/integration tests. No downloads or large model weights."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from aquaroute import cli
from aquaroute.config import ROOT, load_config
from aquaroute.data import expand_conditions, load_jsonl, read_samples
from aquaroute.engine import infer
from aquaroute.metrics import evaluate
from aquaroute.protocol import parse_prediction, validate_truth
from aquaroute.storage import PredictionStore, run_lock

ANSWER = '{"images":[{"image_index":0,"present":true}]}'


class FakeModel:
    def __init__(self, *, max_batch=100, interrupt_after=None, bad_ids=()):
        self.calls = []
        self.max_batch = max_batch
        self.interrupt_after = interrupt_after
        self.bad_ids = bad_ids

    def predict_batch(self, requests):
        self.calls.append(len(requests))
        if self.interrupt_after is not None and len(self.calls) > self.interrupt_after:
            raise KeyboardInterrupt()
        if len(requests) > self.max_batch:
            raise RuntimeError("CUDA out of memory")
        return ['broken' if Path(paths[0]).name.split('_')[0] in self.bad_ids else
                json.dumps({"images": [{"image_index": i, "present": True} for i in range(len(paths))]})
                for paths, _, _ in requests]

    def predict(self, paths, question, target):
        return self.predict_batch([(paths, question, target)])[0]


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = load_config('config/recognition.yaml')
        self.cfg['model']['path'] = str(self.root/'model')
        self.cfg['inference']['batch_size'] = 4

    def prepared(self, n=5):
        rows = []
        for i in range(n):
            conditions = {}
            for condition in ['original', 'gamma_Low']:
                path = self.root / f'{i}_{condition}.png'
                Image.new('RGB', (8, 8), (i, 40, 60)).save(path)
                conditions[condition] = [str(path)]
            rows.append({'id': str(i), 'target': '海胆', 'question': '存在吗',
                         'truth': [{'count': 1}], 'conditions': conditions})
        return rows

    def tasks(self, n=5):
        return expand_conditions(self.prepared(n), self.cfg)


class ProtocolTests(unittest.TestCase):
    def test_presence(self):
        self.assertTrue(parse_prediction(ANSWER, 1)['images'][0]['present'])

    def test_fences_and_surrounding_text(self):
        self.assertEqual(parse_prediction('```json\n'+ANSWER+'\n```', 1), json.loads(ANSWER))
        self.assertEqual(parse_prediction('answer: '+ANSWER, 1), json.loads(ANSWER))

    def test_invalid_shapes_and_types(self):
        for value in [[], {}, {'images': []}, {'images': [{'image_index': True, 'present': True}]},
                      {'images': [{'image_index': 0, 'present': 'true'}]},
                      {'images': [{'image_index': 0, 'present': True, 'count': 1}]}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_prediction(json.dumps(value), 1)

    def test_classification_unknown_and_closed_labels(self):
        raw = '{"images":[{"image_index":0,"label":"unknown"}]}'
        self.assertEqual(parse_prediction(raw, 1, 'classification', ['海胆'])['images'][0]['label'], 'unknown')
        with self.assertRaises(ValueError):
            parse_prediction(raw.replace('unknown', 'wrong'), 1, 'classification', ['海胆'])

    def test_count_truth_compatibility_and_conflicts(self):
        validate_truth([{'count': 0}, {'count': 3}], 2, 'presence')
        for truth in [[{'count': -1}], [{'present': 1}], [{'count': 1, 'present': False}], [{}]]:
            with self.subTest(truth=truth), self.assertRaises(ValueError):
                validate_truth(truth, 1, 'presence')


class DataTests(Fixture):
    def test_manifest_relative_paths_and_count_truth(self):
        path = self.root/'image.png'; Image.new('RGB', (4,4)).save(path)
        manifest = self.root/'manifest.jsonl'
        manifest.write_text(json.dumps({'id':'a','images':['image.png'],'question':'q','target':'t','truth':[{'count':1}]})+'\n',encoding='utf-8')
        self.assertEqual(read_samples(manifest, self.cfg)[0]['images'], [str(path.resolve())])

    def test_task_order_and_condition_independence(self):
        tasks = self.tasks(2)
        self.assertEqual([(r['id'],r['condition']) for r in tasks], [('0','original'),('0','gamma_Low'),('1','original'),('1','gamma_Low')])

    def test_reject_duplicate_ids_and_missing_files(self):
        prepared = self.prepared(2); prepared[1]['id'] = '0'
        with self.assertRaises(ValueError): expand_conditions(prepared, self.cfg)
        prepared = self.prepared(1); Path(prepared[0]['conditions']['original'][0]).unlink()
        with self.assertRaises(FileNotFoundError): expand_conditions(prepared, self.cfg)

    def test_reject_mixed_images_and_truth_lengths(self):
        prepared = self.prepared(1)
        prepared[0]['conditions']['original'] = prepared[0]['conditions']['gamma_Low']
        with self.assertRaises(ValueError): expand_conditions(prepared, self.cfg)
        prepared = self.prepared(1); prepared[0]['truth'] = []
        with self.assertRaises(ValueError): expand_conditions(prepared, self.cfg)

    def test_explicit_path_map(self):
        prepared = self.prepared(1)
        for condition, paths in prepared[0]['conditions'].items():
            paths[0] = '/old/server/' + Path(paths[0]).name
        tasks = expand_conditions(prepared, self.cfg, [('/old/server', str(self.root))])
        self.assertTrue(all(Path(p).is_file() for r in tasks for p in r['images']))

    def test_include_original_false(self):
        self.cfg['inference']['include_original'] = False
        self.assertEqual([r['condition'] for r in self.tasks(2)], ['gamma_Low','gamma_Low'])

    def test_validation_cli_does_not_load_model(self):
        path = self.root/'prepared.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in self.prepared(1)),encoding='utf-8')
        with patch.object(cli,'load_config',return_value=self.cfg), contextlib.redirect_stdout(io.StringIO()) as output:
            code=cli.run(cli.parser().parse_args(['validate','--prepared',str(path)]))
        self.assertEqual(code,0);self.assertIn('2 conditions',output.getvalue())


class EngineTests(Fixture):
    def test_batches_and_output_schema(self):
        tasks = self.tasks(); model = FakeModel()
        records = infer(tasks, self.cfg, self.root/'run', model_factory=lambda:model)
        self.assertEqual(model.calls, [4,4,2])
        self.assertEqual([(r['id'],r['condition']) for r in records], [(r['id'],r['condition']) for r in tasks])
        self.assertTrue(all(r['error'] is None for r in records))
        self.assertEqual(load_jsonl(self.root/'run/predictions.jsonl'), records)

    def test_oom_adapts_and_preserves_order(self):
        tasks = self.tasks(); model = FakeModel(max_batch=2)
        records = infer(tasks,self.cfg,self.root/'run',model_factory=lambda:model)
        self.assertEqual(model.calls,[4,2,2,2,2,2])
        self.assertEqual([r['id'] for r in records],[r['id'] for r in tasks])

    def test_parse_failures_are_not_absence(self):
        records = infer(self.tasks(2),self.cfg,self.root/'run',model_factory=lambda:FakeModel(bad_ids=('1',)))
        self.assertEqual(sum(r['prediction'] is None for r in records),2)
        self.assertTrue(all(r['raw']=='broken' and r['error'] for r in records if r['id']=='1'))

    def test_limit_resume_and_no_duplicate_predictions(self):
        tasks=self.tasks(); run=self.root/'run'
        infer(tasks,self.cfg,run,limit=3,model_factory=FakeModel)
        model=FakeModel()
        records=infer(tasks,self.cfg,run,resume=True,model_factory=lambda:model)
        self.assertEqual(sum(model.calls),7)
        self.assertEqual(len(records),10)
        self.assertEqual(len(load_jsonl(run/'predictions.jsonl')),10)

    def test_interrupt_preserves_completed_batch(self):
        tasks=self.tasks(); run=self.root/'run'
        with self.assertRaises(KeyboardInterrupt):
            infer(tasks,self.cfg,run,model_factory=lambda:FakeModel(interrupt_after=1))
        self.assertEqual(len(load_jsonl(run/'predictions.jsonl')),4)
        self.assertEqual(json.loads((run/'run.json').read_text())['state'],'interrupted')
        self.assertEqual(len(infer(tasks,self.cfg,run,resume=True,model_factory=FakeModel)),10)

    def test_completed_resume_does_not_load_model(self):
        tasks=self.tasks(1);run=self.root/'run'
        infer(tasks,self.cfg,run,model_factory=FakeModel)
        def forbidden(): raise AssertionError('Model should not load')
        self.assertEqual(len(infer(tasks,self.cfg,run,resume=True,model_factory=forbidden)),2)

    def test_collision_and_input_changes_rejected(self):
        tasks=self.tasks(1);run=self.root/'run'
        infer(tasks,self.cfg,run,model_factory=FakeModel)
        with self.assertRaises(ValueError): infer(tasks,self.cfg,run,model_factory=FakeModel)
        changed=copy.deepcopy(tasks);changed[0]['target']='changed'
        with self.assertRaises(ValueError): infer(changed,self.cfg,run,resume=True,model_factory=FakeModel)

    def test_batch_size_can_change_on_resume(self):
        tasks=self.tasks();run=self.root/'run'
        infer(tasks,self.cfg,run,limit=4,model_factory=FakeModel)
        self.cfg['inference']['batch_size']=2
        model=FakeModel()
        infer(tasks,self.cfg,run,resume=True,model_factory=lambda:model)
        self.assertEqual(model.calls,[2,2,2])

    def test_partial_tail_recovered_complete_corruption_rejected(self):
        tasks=self.tasks(2);run=self.root/'run'
        infer(tasks,self.cfg,run,limit=2,model_factory=FakeModel)
        with (run/'predictions.jsonl').open('ab') as stream: stream.write(b'{"incomplete":')
        self.assertEqual(len(infer(tasks,self.cfg,run,resume=True,model_factory=FakeModel)),4)
        with (run/'predictions.jsonl').open('ab') as stream: stream.write(b'broken\n')
        with self.assertRaises(ValueError): infer(tasks,self.cfg,run,resume=True,model_factory=FakeModel)

    def test_lock_prevents_second_writer(self):
        run=self.root/'run'
        with run_lock(run):
            with self.assertRaises(RuntimeError):
                with run_lock(run): pass


class MetricsTests(unittest.TestCase):
    def row(self, truth, prediction):
        return {'task':'recognition','recognition_mode':'presence','condition':'original',
                'images':['image'],'truth':[{'present':truth}],'prediction':prediction}

    def test_invalid_answers_and_denominators(self):
        true=json.loads(ANSWER); false=json.loads(ANSWER.replace('true','false'))
        rows=[self.row(True,true),self.row(True,false),self.row(False,false),self.row(False,true),self.row(True,None)]
        m=evaluate(rows)['original']
        self.assertEqual(m['confusion_matrix'],{'tn':1,'fp':1,'fn':1,'tp':1,'invalid':1})
        self.assertEqual(m['accuracy'],.4); self.assertEqual(m['recall'],.5)
        self.assertAlmostEqual(m['recall_all'],1/3)
        self.assertEqual(m['majority_baseline'],.6)
        self.assertEqual(m['invalid_positive'],1)

    def test_unlabeled_results(self):
        row=self.row(True,json.loads(ANSWER));row['truth']=None
        m=evaluate([row])['original']
        self.assertIsNone(m['accuracy']);self.assertIsNone(m['majority_baseline'])

    def test_classification_unknown_and_invalid(self):
        def row(truth,pred):
            return {'task':'recognition','recognition_mode':'classification','condition':'original',
                    'images':['x'],'truth':[{'label':truth}],
                    'prediction':{'images':[{'image_index':0,'label':pred}]} if pred else None}
        m=evaluate([row('a','a'),row('a','unknown'),row('b',None)],'classification',['a','b'])['original']
        self.assertAlmostEqual(m['accuracy'],1/3)
        self.assertEqual(m['invalid_by_class'],{'a':0,'b':1})
        self.assertAlmostEqual(m['per_class']['a']['recall_all'],.5)
        self.assertAlmostEqual(m['macro_f1_all'],1/3)

    def test_wrong_task_is_rejected(self):
        row=self.row(True,None);row['task']='count'
        with self.assertRaises(ValueError): evaluate([row])


class CompatibilityTests(unittest.TestCase):
    def test_default_configuration_is_target_presence(self):
        cfg=load_config('config/recognition.yaml')
        self.assertEqual(cfg['task'],'recognition')
        self.assertEqual(cfg['recognition']['mode'],'presence')
        self.assertEqual(cfg['model']['repo_id'],'Qwen/Qwen3.5-9B')
        self.assertEqual(cfg['inference']['batch_size'],8)

    def test_protected_file_hashes(self):
        hashes=json.loads((ROOT/'docs/preserved_sha256.json').read_text(encoding='utf-8'))
        for name,expected in hashes.items():
            with self.subTest(name=name): self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),expected)

    def test_all_new_configurations(self):
        for name in ['count','recognition','classification']:
            with self.subTest(name=name):
                cfg=load_config('config/'+name+'.yaml')
                self.assertTrue((ROOT/cfg['enhancement']['waternet_weights']).is_file())

    def test_legacy_count_evaluation_unchanged(self):
        from mama.evaluation import evaluate as count_evaluate
        rows=[{'condition':'original','truth':[{'count':2}],
               'prediction':{'images':[{'image_index':0,'present':True,'count':2,'locations':['a','b']}]}}]
        m=count_evaluate(rows)['original']
        self.assertEqual(m['count_exact_rate'],1);self.assertEqual(m['count_mae'],0)

    def test_help_works_without_model(self):
        output=subprocess.run([sys.executable,'-m','aquaroute','--help'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(output.returncode,0,output.stderr)
        self.assertIn('--resume',output.stdout)

    def test_count_dispatch_forwards_original_flags(self):
        args=cli.parser().parse_args(['infer','--task','count','--prepared','p.jsonl','--run-id','x'])
        import mama.__main__
        recorded=[]
        with patch.object(mama.__main__,'main',side_effect=lambda:recorded.append(sys.argv.copy())):
            cli.legacy_count(args,'config/count.yaml')
        self.assertEqual(recorded[0][1],'infer');self.assertIn('--prepared',recorded[0]);self.assertIn('x',recorded[0])

    def test_count_rejects_new_runtime_flags(self):
        args=cli.parser().parse_args(['infer','--task','count','--batch-size','8'])
        with self.assertRaises(ValueError): cli.legacy_count(args,'config/count.yaml')


if __name__ == '__main__': unittest.main()
