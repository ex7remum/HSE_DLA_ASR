import os
import gdown

if __name__ == "__main__":
    model_url = 'https://drive.google.com/uc?export=download&id=1KYa__v68ofjDiMRJB4UzZBizC-zi5cIk'
    model_path = 'best_model/model.pth'
    if not os.path.exists(model_path):
        os.mkdir('best_model')
        print('Downloading DeepSpeech2 model.')
        gdown.download(model_url, model_path)
        print('Downloaded DeepSpeech2 model.')
    else:
        print('Deepspeech2 model already exists.')

    config_url = 'https://drive.google.com/uc?export=download&id=1neW1hfHfJ1XNdJ4ztWHSAE62_GFdkAkm'
    config_path = 'best_model/config.json'
    if not os.path.exists(config_path):
        print('Downloading DeepSpeech2 model test config.')
        gdown.download(config_url, config_path)
        print('Downloaded DeepSpeech2 model test config.')
    else:
        print('Deepspeech2 model config already exists.')
