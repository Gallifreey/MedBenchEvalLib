import time
from abc import abstractmethod, ABC

import regex as re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge.rouge import Rouge
from typing import Dict, List

from medeval.lib.metric_base import MeanMetric
from medeval.lib.utils import write_json_data


class LexicalBase(ABC):
    def __init__(self, debug=False, file_name="score"):
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.debug = debug
        self.file_name = file_name
        self.keys = []
        self.logs = []

    def tokenize(self, input_str):
        words = re.findall(self.PAT, input_str)
        return [i.strip() for i in words if len(i.strip()) > 0]

    def debug_print(self, score_dict: Dict[str, MeanMetric]):
        return {k: v.compute()["value"] for k, v in score_dict.items()}

    @abstractmethod
    def compute(self, pred, gt):
        pass

    def batch_compute(self, preds, gts):
        for pred, gt in zip(preds, gts):
            total, cur, meta = self.compute(pred, gt)
            if self.debug:
                self.logs.append({
                    "pred": meta["pred"],
                    "gt": meta["gt"],
                    "current_score": cur,
                    "total_score": total
                })
        write_json_data(f"{self.file_name}-{time.time()}.json", self.logs)
        self.logs = []


class BLEUMetric(LexicalBase):
    def __init__(self, debug=False):
        super().__init__(debug=debug)
        self.keys = ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4', 'bleu-mean']
        self.score_dict = {key: MeanMetric() for key in self.keys}

    def compute(self, pred, gt):
        hypothesis, reference = self.tokenize(pred), self.tokenize(gt)
        bleu1_score = sentence_bleu([reference], hypothesis, weights=(1., 0, 0, 0),
                                    smoothing_function=SmoothingFunction().method1)
        bleu2_score = sentence_bleu([reference], hypothesis, weights=(0, 1., 0, 0),
                                    smoothing_function=SmoothingFunction().method1)
        bleu3_score = sentence_bleu([reference], hypothesis, weights=(0, 0, 1., 0),
                                    smoothing_function=SmoothingFunction().method1)
        bleu4_score = sentence_bleu([reference], hypothesis, weights=(0, 0, 0, 1.),
                                    smoothing_function=SmoothingFunction().method1)
        bleu_mean_score = (bleu1_score + bleu2_score + bleu3_score + bleu4_score) / 4.

        self.score_dict['bleu-1'].update(bleu1_score)
        self.score_dict['bleu-2'].update(bleu2_score)
        self.score_dict['bleu-3'].update(bleu3_score)
        self.score_dict['bleu-4'].update(bleu4_score)
        self.score_dict['bleu-mean'].update(bleu_mean_score)

        return {
                   "bleu-1": self.score_dict['bleu-1'].compute()["value"],
                   "bleu-2": self.score_dict['bleu-2'].compute()["value"],
                   "bleu-3": self.score_dict['bleu-3'].compute()["value"],
                   "bleu-4": self.score_dict['bleu-4'].compute()["value"],
                   "bleu-mean": self.score_dict['bleu-mean'].compute()["value"]
               }, {
                   "bleu-1": bleu1_score,
                   "bleu-2": bleu2_score,
                   "bleu-3": bleu3_score,
                   "bleu-4": bleu4_score,
                   "bleu-mean": bleu_mean_score
               }, {
                   "pred": pred,
                   "gt": gt
               }


class ROUGEMetric(LexicalBase):
    def __init__(self, debug=False):
        super().__init__(debug=debug)
        self.keys = ['rouge-1', 'rouge-2', 'rouge-l']
        self.score_dict = {key: MeanMetric() for key in self.keys}

    def compute(self, pred, gt):
        rouge = Rouge()
        scores = rouge.get_scores(pred, gt)[0]
        rouge_1_score = scores["rouge-1"]['f']
        rouge_2_score = scores["rouge-2"]['f']
        rouge_l_score = scores["rouge-l"]['f']
        self.score_dict['rouge-1'].update(rouge_1_score)
        self.score_dict['rouge-2'].update(rouge_2_score)
        self.score_dict['rouge-l'].update(rouge_l_score)

        return {
                   "rouge-1": self.score_dict['rouge-1'].compute()["value"],
                   "rouge-2": self.score_dict['rouge-2'].compute()["value"],
                   "rouge-l": self.score_dict['rouge-l'].compute()["value"]
               }, {
                   "rouge-1": rouge_1_score,
                   "rouge-2": rouge_2_score,
                   "rouge-l": rouge_l_score
               }, {
                   "pred": pred,
                   "gt": gt
               }


class METEORMetric(LexicalBase):
    def __init__(self, debug=False):
        super().__init__(debug=debug)
        self.keys = ['meteor']
        self.score_dict = {key: MeanMetric() for key in self.keys}

    def compute(self, pred, gt):
        hypothesis, reference = self.tokenize(pred), self.tokenize(gt)
        meteor = meteor_score([reference], hypothesis)
        self.score_dict['meteor'].update(meteor)

        return {
                   "meteor": self.score_dict['meteor'].compute()["value"]
               }, {
                   "meteor": meteor
               }, {
                   "pred": pred,
                   "gt": gt
               }


class MixMetric(LexicalBase):
    def __init__(self, metric_list: List[LexicalBase], debug=False):
        super().__init__(debug=debug)
        self.keys = []
        for metric in metric_list:
            self.keys.extend(metric.keys)
        self.score_dict = {key: MeanMetric() for key in self.keys}

    def compute(self, pred, gt):
        pass


bleu = METEORMetric(debug=True)
bleu.batch_compute(["""
 In comparison with study of there is
 again enlargement of the cardiac silhouette with a pacer device in place. No definite vascular congestion
 raising the possibility of underlying
 cardiomyopathy or pleural effusion.
 No acute focal pneumonia. The right picc line has been removed.
""", """
 There are low lung volumes. Bibasilar atelectasis have minimally im
proved. Mild vascular congestion
 has minimally improved. There are
 no new lung abnormalities or pneumothorax. Bilateral pleural effusions are small. Right picc tip is at the
cavoatrial junction.
"""], ["""
In comparison with the study of there is little change in the appearance
 of the pacer leads which extend to
 the right atrium and apex of the right
 ventricle. Continued enlargement of
 the cardiac silhouette without vascular congestion or pleural effusion.
 No evidence of pneumothorax.
""", """
The lung volumes are low. There is
 a small left pleural effusion with associated atelectasis. The right lung
 is clear. There is no pneumothorax.
 The heart size is top normal. The hi
lar and mediastinal contours are normal. A right subclavian catheter terminate in the mid svc.
"""])
