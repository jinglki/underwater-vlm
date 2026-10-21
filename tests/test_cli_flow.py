"""End-to-end CLI orchestration using real classic enhancements and a fake model.

Only Zero-DCE and WaterNet GPU execution is replaced during this test. This is
not validation of those neural enhancement algorithms or actual Qwen inference.
"""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

from aquaroute.cli import parser, run
from aquaroute.config import load_config
from aquaroute.data import load_jsonl


class Model:
    def __init__(self,*a): pass
    def predict(self,paths,q,t):
        return json.dumps({'images':[{'image_index':i,'present':True} for i in range(len(paths))]})
    def predict_batch(self,requests): return [self.predict(*r) for r in requests]


class CliFlowTests(unittest.TestCase):
    def test_prepare_infer_resume_evaluate(self):
        import mama.enhancements as enhancements
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            cfg=load_config('config/recognition.yaml')
            cfg['model']['path']=str(root/'model')
            cfg['enhancement']['output_dir']=str(root/'enhanced')
            cfg['inference']['output_dir']=str(root/'runs')
            cfg['evaluation']['output_dir']=str(root/'results')
            cfg['preprocess']['width']=32;cfg['preprocess']['height']=32
            config=root/'config.yaml';config.write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
            image=root/'source.png';Image.new('RGB',(32,32),(40,70,90)).save(image)
            manifest=root/'manifest.jsonl'
            manifest.write_text(json.dumps({'id':'test','images':['source.png'],'target':'海胆','question':'存在吗',
                                            'truth':[{'present':True}]},ensure_ascii=False)+'\n',encoding='utf-8')
            with patch.object(enhancements,'_waternet',side_effect=lambda rgb,cfg:rgb.copy()), \
                 patch.object(enhancements,'_zero_dce',side_effect=lambda rgb,cfg:rgb.copy()):
                self.assertEqual(run(parser().parse_args(['prepare','--config',str(config),'--manifest',str(manifest),'--run-id','prepared'])),0)
            prepared=root/'runs/prepared/prepared.jsonl'
            self.assertEqual(len(load_jsonl(prepared)[0]['conditions']),25)
            backend=types.ModuleType('aquaroute.model');backend.VisionModel=Model
            base=['--config',str(config),'--prepared',str(prepared),'--run-id','infer']
            with patch.dict('sys.modules',{'aquaroute.model':backend}):
                self.assertEqual(run(parser().parse_args(['infer',*base,'--limit','16'])),0)
                predictions=root/'runs/infer/predictions.jsonl'
                self.assertEqual(len(load_jsonl(predictions)),16)
                with self.assertRaises(ValueError):
                    run(parser().parse_args(['evaluate','--config',str(config),'--predictions',str(predictions),'--run-id','infer']))
                self.assertEqual(run(parser().parse_args(['infer',*base,'--resume'])),0)
            self.assertEqual(len(load_jsonl(predictions)),25)
            self.assertEqual(run(parser().parse_args(['evaluate','--config',str(config),'--predictions',str(predictions),'--run-id','infer'])),0)
            report=json.loads((root/'results/infer_metrics.json').read_text())
            self.assertEqual(len(report),25)
            self.assertTrue(all(v['accuracy']==1 for v in report.values()))
            changed=copy.deepcopy(cfg);changed['recognition']={'mode':'classification','labels':['海胆']}
            changed_path=root/'changed.yaml';changed_path.write_text(yaml.safe_dump(changed,allow_unicode=True),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'mode/labels'):
                run(parser().parse_args(['evaluate','--config',str(changed_path),'--predictions',str(predictions),'--run-id','infer']))


if __name__=='__main__': unittest.main()
