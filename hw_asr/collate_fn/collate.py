import logging
from typing import List
import torch
import numpy as np
from hw_asr.text_encoder.char_text_encoder import CharTextEncoder

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    raw_texts = [item['text'] for item in dataset_items]
    raw_encoded_texts = [item['text_encoded'] for item in dataset_items]
    text_lengths = np.array([text.shape[1] for text in raw_encoded_texts])
    max_len_encoded = text_lengths.max()

    batch_size = len(raw_texts)
    encoded_texts_padded = torch.zeros((batch_size, max_len_encoded))
    for item_num, encoded_text in enumerate(raw_encoded_texts):
        encoded_texts_padded[item_num, :encoded_text.shape[1]] = encoded_text[0]

    spectrogram_lengths = np.array([item['spectrogram'].shape[2] for item in dataset_items])
    max_spec_time = spectrogram_lengths.max()
    feature_length_dim = dataset_items[0]['spectrogram'].shape[1]
    spectrogram = torch.zeros((batch_size, feature_length_dim, max_spec_time))
    for item_num, item in enumerate(dataset_items):
        cur_spec = item['spectrogram']
        spectrogram[item_num, :, :cur_spec.shape[2]] = cur_spec[0]

    paths = [item["audio_path"] for item in dataset_items]

    result_batch = {'text': raw_texts,
                    'text_encoded': encoded_texts_padded,
                    'text_encoded_length': torch.tensor(text_lengths),
                    'spectrogram': spectrogram,
                    'spectrogram_length': torch.tensor(spectrogram_lengths),
                    'audio_path': paths}
    return result_batch
