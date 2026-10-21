"""Backend contract tests with simulated torch tensors and real small PIL images."""
import contextlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image
from aquaroute.config import ROOT


class Tensor:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))
    def is_floating_point(self): return False
    def to(self, *args, **kwargs): return self
    def __getitem__(self, key): return Tensor([row[key[1]] for row in self.rows])


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'image.png'; Image.new('RGB',(8,8),(10,20,30)).save(self.path)
        torch = types.ModuleType('torch'); torch.Tensor = Tensor; torch.inference_mode = contextlib.nullcontext
        transformers = types.ModuleType('transformers')
        transformers.AutoProcessor = object; transformers.AutoModelForImageTextToText = object
        self.modules = patch.dict(sys.modules, {'torch':torch,'transformers':transformers})
        self.modules.start(); self.addCleanup(self.modules.stop)
        spec = importlib.util.spec_from_file_location('aquaroute._backend_test', ROOT/'aquaroute/model.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.vision = module.VisionModel.__new__(module.VisionModel)
        self.vision.kind = 'qwen35'; self.vision.recognition={'mode':'presence'}; self.vision.image_workers=4
        self.vision.device='cuda'; self.vision.dtype=None; self.vision.max_new_tokens=128
        self.calls=[]
        self.vision.model=types.SimpleNamespace(generate=self.generate)

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return Tensor([row+[8,9] for row in kwargs['input_ids'].rows])

    def processor(self, modern=True, fail=False):
        captured=[]
        class Common:
            tokenizer=types.SimpleNamespace(padding_side='right',pad_token_id=0,eos_token_id=1)
            def encode(self, messages, options, kwargs):
                if fail: raise RuntimeError('processor failed')
                is_batch=options.get('padding',False)
                conversations=messages if is_batch else [messages]
                for conversation in conversations:
                    for content in conversation[-1]['content']:
                        if content['type']=='image': content['image'].getpixel((0,0))
                captured.append((conversations,options,kwargs,self.tokenizer.padding_side))
                return {'input_ids':Tensor([[1,2,3]]*len(conversations))}
            def batch_decode(self, tokens, **kwargs):
                self.decoded=tokens.rows
                return ['answer-'+str(i) for i in range(len(tokens.rows))]
        class Modern(Common):
            def apply_chat_template(self, messages, processor_kwargs=None, **kwargs):
                if 'padding' in kwargs: raise AssertionError('padding leaked outside processor_kwargs')
                return self.encode(messages,processor_kwargs or {},kwargs)
        class Old(Common):
            def apply_chat_template(self, messages, **kwargs):
                return self.encode(messages,{'padding':kwargs.get('padding',False)},kwargs)
        self.vision.processor=(Modern if modern else Old)()
        return captured

    def test_modern_kwargs_image_order_left_padding_and_prompt_crop(self):
        captured=self.processor()
        requests=[([str(self.path)],'Q1','T1'),([str(self.path),str(self.path)],'Q2','T2')]
        self.assertEqual(self.vision.predict_batch(requests),['answer-0','answer-1'])
        conversations,options,kwargs,padding=captured[0]
        self.assertEqual([len(c[-1]['content']) for c in conversations],[2,3])
        self.assertIn('Q1',conversations[0][-1]['content'][-1]['text'])
        self.assertNotIn('Q2',conversations[0][-1]['content'][-1]['text'])
        self.assertEqual(options,{'padding':True});self.assertFalse(kwargs['enable_thinking'])
        self.assertEqual(padding,'left');self.assertEqual(self.vision.processor.tokenizer.padding_side,'right')
        self.assertEqual(self.vision.processor.decoded,[[8,9],[8,9]])
        self.assertFalse(self.calls[0]['do_sample']);self.assertEqual(self.calls[0]['max_new_tokens'],128)
        for c in conversations:
            with self.assertRaises(ValueError): c[-1]['content'][0]['image'].getpixel((0,0))

    def test_older_transformers_contract(self):
        captured=self.processor(modern=False)
        self.vision.predict_batch([([str(self.path)],'q','t')]*2)
        self.assertTrue(captured[0][1]['padding'])

    def test_single_request_public_interface(self):
        captured=self.processor()
        self.assertEqual(self.vision.predict([str(self.path)],'q','t'),'answer-0')
        self.assertFalse(captured[0][1].get('padding',False))

    def test_padding_restored_on_processing_failure(self):
        self.processor(fail=True)
        with self.assertRaises(RuntimeError): self.vision.predict_batch([([str(self.path)],'q','t')]*2)
        self.assertEqual(self.vision.processor.tokenizer.padding_side,'right')

    def test_classification_does_not_use_count_protocol(self):
        captured=self.processor();self.vision.recognition={'mode':'classification','labels':['海胆','海星']}
        self.vision.predict([str(self.path)],'q','t')
        conversations=captured[0][0]
        self.assertIn('候选类别',conversations[0][-1]['content'][-1]['text'])
        self.assertIn('海星',conversations[0][-1]['content'][-1]['text'])
        self.assertIn('"label"',conversations[0][0]['content'])

    def test_empty_batch_and_empty_image_request(self):
        self.assertEqual(self.vision.predict_batch([]),[])
        with self.assertRaises(ValueError): self.vision.predict([],'q','t')

    def test_missing_pad_token_is_explicit_and_restores_state(self):
        self.processor(); self.vision.processor.tokenizer.pad_token_id=None
        with self.assertRaisesRegex(ValueError,'pad token'):
            self.vision.predict_batch([([str(self.path)],'q','t')]*2)
        self.assertEqual(self.vision.processor.tokenizer.padding_side,'right')


if __name__=='__main__': unittest.main()
