import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from transformers import AutoImageProcessor, ViTForImageClassification, ViTImageProcessor, ViTFeatureExtractor
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
from PIL import Image
from transformers import BertTokenizer, BertForSequenceClassification
import re
from transformers import AdamW


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_vision_transformer_model():
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    # feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        output_hidden_states = True
        )
    model.to(device)
    return model, image_processor


def load_BERT_model(bert_base_uncased, device):
    model = BertForSequenceClassification.from_pretrained(
        bert_base_uncased, # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # Only two classes, on-topic and off-topic  
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = True, # Whether the model returns all hidden-states.
        )
    if device == 'cuda':
        model.to(device)
    return model

def load_trained_BERT_model(args):
    model = torch.load(args.model_path)
    model.eval().to(device)
    return model


def load_optimiser(model, args):
    optimizer = AdamW(model.base_model.parameters(),
                    lr = args.learning_rate,
                    eps = args.adam_epsilon,
                    no_deprecation_warning=True
                    # weight_decay = 0.01
                    )
    return optimizer