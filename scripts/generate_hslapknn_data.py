import os

os.system('python MetaWSD/train_wsd.py --config MetaWSD/config/wsd/hslapknn/hslapknn_bert_4_cuda.yaml')
os.system('python MetaWSD/train_wsd.py --config MetaWSD/config/wsd/hslapknn/hslapknn_bert_8_cuda.yaml')
os.system('python MetaWSD/train_wsd.py --config MetaWSD/config/wsd/hslapknn/hslapknn_bert_16_cuda.yaml')
os.system('python MetaWSD/train_wsd.py --config MetaWSD/config/wsd/hslapknn/hslapknn_bert_32_cuda.yaml')
os.system('python MetaWSD/train_wsd.py --config MetaWSD/config/wsd/hslapknn/hslapknn_elmo_4_cuda.yaml')
os.system('python MetaWSD/train_wsd.py --config MetaWSD/config/wsd/hslapknn/hslapknn_elmo_8_cuda.yaml')
os.system('python MetaWSD/train_wsd.py --config MetaWSD/config/wsd/hslapknn/hslapknn_elmo_16_cuda.yaml')
os.system('python MetaWSD/train_wsd.py --config MetaWSD/config/wsd/hslapknn/hslapknn_elmo_32_cuda.yaml')
