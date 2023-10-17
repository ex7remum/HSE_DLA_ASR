import gzip
import os, shutil, wget

# from official tutorial
# https://github.com/kensho-technologies/pyctcdecode/blob/main/tutorials/01_pipeline_nemo.ipynb
if __name__ == "__main__":
    # getting LM
    lm_gzip_path = '3-gram.pruned.1e-7.arpa.gz'
    if not os.path.exists(lm_gzip_path):
        print('Downloading pruned 3-gram model.')
        lm_url = 'http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz'
        lm_gzip_path = wget.download(lm_url)
        print('Downloaded the 3-gram language model.')
    else:
        print('Pruned .arpa.gz already exists.')

    uppercase_lm_path = '3-gram.pruned.1e-7.arpa'
    if not os.path.exists(uppercase_lm_path):
        with gzip.open(lm_gzip_path, 'rb') as f_zipped:
            with open(uppercase_lm_path, 'wb') as f_unzipped:
                shutil.copyfileobj(f_zipped, f_unzipped)
        print('Unzipped the 3-gram language model.')
    else:
        print('Unzipped .arpa already exists.')

    lm_path = 'lowercase_3-gram.pruned.1e-7.arpa'
    if not os.path.exists(lm_path):
        with open(uppercase_lm_path, 'r') as f_upper:
            with open(lm_path, 'w') as f_lower:
                for line in f_upper:
                    f_lower.write(line.lower())
    print('Converted language model file to lowercase.')

    # getting vocabulary
    vocab_path = 'librispeech-vocab.txt'
    if not os.path.exists(vocab_path):
        vocab_url = 'http://www.openslr.org/resources/11/librispeech-vocab.txt'
        vocab_path = wget.download(vocab_url)
        print('Downloaded unigram vocab.')
    else:
        print('Unigram vocab already exists.')
