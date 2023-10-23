from typing import List, NamedTuple
from collections import defaultdict
import torch
from pyctcdecode import build_ctcdecoder
import kenlm

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        with open("librispeech-vocab.txt") as f:
            unigram_list = [t.lower() for t in f.read().strip().split("\n")]

        self.lm_decoder = build_ctcdecoder([''] + self.alphabet,
                                           kenlm_model_path='lowercase_3-gram.pruned.1e-7.arpa',
                                           unigrams=unigram_list)

    def ctc_decode(self, inds: List[int]) -> str:
        result = []
        last_char = self.EMPTY_TOK
        EMPTY_IND = self.char2ind[self.EMPTY_TOK]
        for ind in inds:
            if ind == EMPTY_IND:
                last_char = self.EMPTY_TOK
                continue
            if last_char != self.ind2char[ind]:
                result.append(self.ind2char[ind])
            last_char = self.ind2char[ind]
        return ''.join(result)
    
    def extend_and_merge(self, frame, state):
        new_state = defaultdict(float)
        for next_char_index, next_char_proba in enumerate(frame):
            for (pref, last_char), pref_proba in state.items():
                next_char = self.ind2char[next_char_index]
                if next_char == last_char:
                    new_pref = pref
                else:
                    if next_char != self.EMPTY_TOK:
                        new_pref = pref + next_char
                    else:
                        new_pref = pref
                    last_char = next_char
                new_state[(new_pref, last_char)] += pref_proba * next_char_proba
        return new_state
    
    def truncate(self, state, beam_size):
        state_list = list(state.items())
        state_list.sort(key=lambda x: -x[1])
        return dict(state_list[:beam_size])
    
    def ctc_beam_search(self, probs: torch.tensor,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []
        
        state = {('', self.EMPTY_TOK) : 1.0}
        for frame in probs:
            state = self.extend_and_merge(frame, state)
            state = self.truncate(state, beam_size)
        state_list = list(state.items())
        hypos = [Hypothesis(v[0][0], v[-1]) for v in state_list]
        return sorted(hypos, key=lambda x: x.prob, reverse=True)

    def lm_ctc_beam_search(self, probs: torch.tensor,
                           beam_size: int = 100,
                           alpha: float = 0.7,
                           beta: float = 3.) -> List[Hypothesis]:
        self.lm_decoder.reset_params(alpha=alpha, beta=beta)
        logits = torch.logit(probs)
        return [Hypothesis(self.lm_decoder.decode(logits.cpu().numpy(), beam_width=beam_size), 1.0)]

