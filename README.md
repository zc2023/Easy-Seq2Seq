# Seq2Seq-EasyStartup-Pytorch
This repository provides an easy-to-deploy Seq2Seq model for NLP beginners using the latest (2024) PyTorch version. It includes implementations for both traditional Seq2Seq models and more advanced architectures like Transformer and RNN+Attention for English-Chinese translation.

## Keywords
To make this repository easy to search, the following keywords are included:

Seq2SeqModel, Translation, NLP, Chinese2English, English2Chinese
英译中模型, 中译英模型, 英汉互译, 英文翻译中文，中文翻译英文，
pytorch, en2cn, cn2en, pytorch
LSTM, 基于RNN的Seq2Seq（无Attention）英汉互译
基于RNN+Attention的Seq2Seq英汉互译, 基于Transformer的英汉互译
## Getting Started
1. Install Anaconda
Follow the steps below to install dependencies and set up your environment.
2. Create Environment and Install Dependencies 
Create a new Conda environment and install the necessary packages:
```
conda create --name ml python=3.8.19
conda activate ml
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
pip install spacy==3.6.1
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
```

## Main requirements
- torch==2.0.0
- torchtext==0.15.1
- spacy==3.6.1
Full list of dependencies is available in `requirements.txt`.

## Dataset
### Download Dataset
Download Google's Chinese-to-English translation dataset from [here](https://www.kaggle.com/datasets/qianhuan/translation/data). Before using it, you need to preprocess it using `fix.py`.
For quick deployment, you can use the fixed smaller dataset `translation2019zh_valid_fixed.json`.

### Download Pretrained Weights


## Train
1. Set configuration parameters in `train.py`:
```
train(num_epochs=50,
        learning_rate = 0.001, # 0.001
        batch_size = 256,
        load_model_ckpt ='', # whether to use ckpt
        save_model_ckpt_name = 'checkpoint_valid_dataset.pth.tar',
        device = 'cuda:0', # your device, if you don't have gpu, use 'cpu'
        model = "Seq2Seq",
        test_sentence = "你知道的，我会永远爱着你。",
        dataset_path = "YOUR_DATASET_PATH/en2cn/translation2019zh_valid_fixed.json",
        vocab_path = "vocab30k_valid",
        run_name = 'Seq2Seq_100epoch_valid',
        epoch_per_eval = 50,
        epoch_per_save = 25,
        )
```
2. Start training by running:
```
python train.py
```

## Inference
After training, you can use the trained model for inference.
1. Set the configuration in `inference.py`:
```
inference(
       english_vocab_path = "vocab30k/english_vocab.pt",
       chinese_vocab_path = "vocab30k/chinese_vocab.pt",
       model_ckpt_path = "checkpoint_valid_dataset.pth.tar",
       device = "cuda:0",
       model = "Seq2Seq",
       # sentence_to_translate = '美国缓慢地开始倾听，但并非没有艰难曲折。',
       # sentence_to_translate = '你知道的，我会永远爱着你。',
       sentence_to_translate = '昨天有人去超市买了一瓶啤酒',
       # sentence_to_translate = '你好',
)
```
2. Run inference with:
```
python inference.py
```

## Project Structure
### Key Files:
- get_loader.py: Defines the data loader using `torchtext`.
- train.py: Script for training the Seq2Seq model.
- model.py: Defines the Seq2Seq model architecture.
- inference.py: Translates Chinese sentences into English using the trained model.


## Acknowledgment
This project is based on the following open-source repositories:

- [Chinese2English-Translation-seq2seq](https://github.com/Mountchicken/Chinese2English-Translation-seq2seq) 
- [pytorch-seq2seq](https://github.com/bentrevett/pytorch-seq2seq/tree/main). 

Special thanks to [ChatGPT](https://chatgpt.com/) for generating parts of the code, which saved considerable time during development.

