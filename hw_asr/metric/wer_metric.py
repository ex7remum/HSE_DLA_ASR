from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_wer


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            else:
                pred_text = self.text_encoder.decode(log_prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class BeamSearchWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = kwargs['beam_size']

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        wers = []
        lengths = log_probs_length.detach().numpy()
        for pred, length, target_text in zip(log_probs, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            beam_search_result = self.text_encoder.ctc_beam_search(torch.exp(pred[:length]), self.beam_size)
            pred_text = beam_search_result[0].text
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class LMBeamSearchWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = kwargs['beam_size']
        self.alpha = kwargs['alpha']
        self.beta = kwargs['beta']

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        wers = []
        lengths = log_probs_length.detach().numpy()
        for pred, length, target_text in zip(log_probs, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            beam_search_result = self.text_encoder.lm_ctc_beam_search(torch.exp(pred[:length]),
                                                                      self.beam_size, self.alpha, self.beta)
            pred_text = beam_search_result[0].text
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)
