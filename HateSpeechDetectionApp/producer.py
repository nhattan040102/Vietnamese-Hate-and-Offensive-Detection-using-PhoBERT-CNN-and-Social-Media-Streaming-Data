import pprint
import sys
from pathlib import Path
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
import pandas as pd
import numpy as np
from kafka import KafkaProducer
import logging
import socket
import json
import time
import os
import tensorflow as tf
import pickle
from preprocessing import preprocessing
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

# Build service for calling the Youtube API:
## Arguments that need to passed to the build function
DEVELOPER_KEY = "AIzaSyDbt-xdAOjDhJghQGVMxfbsSiSyCFJr1Jw"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

## creating Youtube Resource Object
youtube_service = build(YOUTUBE_API_SERVICE_NAME,
                        YOUTUBE_API_VERSION,
                        developerKey=DEVELOPER_KEY)


class CNN(nn.Module):
        def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):

            super().__init__()

            self.fc_input = nn.Linear(embedding_dim,embedding_dim)

            self.conv_0 = nn.Conv1d(in_channels = embedding_dim,
                                    out_channels = n_filters,
                                    kernel_size = filter_sizes[0])

            self.conv_1 = nn.Conv1d(in_channels = embedding_dim,
                                    out_channels = n_filters,
                                    kernel_size = filter_sizes[1])

            self.conv_2 = nn.Conv1d(in_channels = embedding_dim,
                                    out_channels = n_filters,
                                    kernel_size = filter_sizes[2])

            self.conv_3 = nn.Conv1d(in_channels = embedding_dim,
                                    out_channels = n_filters,
                                    kernel_size = filter_sizes[3])

            self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

            self.dropout = nn.Dropout(dropout)

        def forward(self, encoded):

            #embedded = [batch size, sent len, emb dim]
            embedded = self.fc_input(encoded)
            #print(embedded.shape)

            embedded = embedded.permute(0, 2, 1)
            #print(embedded.shape)

            #embedded = [batch size, emb dim, sent len]

            conved_0 = F.relu(self.conv_0(embedded))
            conved_1 = F.relu(self.conv_1(embedded))
            conved_2 = F.relu(self.conv_2(embedded))
            conved_3 = F.relu(self.conv_3(embedded))

            #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

            pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
            pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
            pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
            pooled_3 = F.max_pool1d(conved_3, conved_3.shape[2]).squeeze(2)

            #pooled_n = [batch size, n_fibatlters]

            cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2, pooled_3), dim = 1))

            #cat = [batch size, n_filters * len(filter_sizes)]

            result =  self.fc(cat)

            #print(result.shape)

            return result
        

# Create a producer
def create_producer():
    try:
        producer = KafkaProducer(bootstrap_servers='localhost:9092')        
    except Exception as e:
        print("Couldn't create the producer")
        producer = None
    return producer


### Function to get youtube video id.
# source:
# https://stackoverflow.com/questions/45579306/get-youtube-video-url-or-youtube-video-id-from-a-string-using-regex
def get_id(url):
    u_pars = urlparse(url)
    quer_v = parse_qs(u_pars.query).get('v')
    if quer_v:
        return quer_v[0]
    pth = u_pars.path.split('/')
    if pth:
        return pth[-1]


def predict_label(text):
    phoBertTokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    device = torch.device('cpu')

    EMBEDDING_DIM = 768
    N_FILTERS = 32
    FILTER_SIZES = [1,2,3,5]
    OUTPUT_DIM = 3
    DROPOUT = 0.1
    PAD_IDX = phoBertTokenizer.pad_token_id
    cnn = CNN(EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

    # load model
    phoBert = torch.load('/Users/macos/Documents/nhattan/CS431/models/phobert_cnn_model_part1_task2a_2.pt', map_location=device)
    cnn = torch.load('/Users/macos/Documents/nhattan/CS431/models/phobert_cnn_model_part2_task2a_2.pt', map_location=device)
    phoBert.eval()
    cnn.eval()

    # processing the text input    
    processed_sentence = preprocessing(text)

    # # Tokenize the sentence using PhoBERT tokenizer
    phobert_inputs = phoBertTokenizer(processed_sentence, return_tensors="pt")


    embedded = phoBert(phobert_inputs['input_ids'], phobert_inputs['attention_mask'])[0]
    predictions = cnn(embedded)
    predictions = predictions.detach().cpu().numpy()
    predictions = np.argmax(predictions,axis=1).flatten()    

    return int(predictions[0])

def main():
    video_link = "https://www.youtube.com/watch?v=lhznO_xsbfU"
    num_comment = 50

    response = youtube_service.commentThreads().list(
        part='snippet',
        maxResults=num_comment,
        textFormat='plainText',
        order='time',
        videoId=get_id(video_link)
    ).execute()

    # create kafka producer
    producer = create_producer()


    try:        
            results = response.get('items', [])
            for item in results:
                author = item['snippet']['topLevelComment']['snippet']['authorDisplayName']            
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                datetime = item['snippet']['topLevelComment']['snippet']['updatedAt']

                # Assume you have your DNN prediction logic here
                pred = predict_label(comment)

                record = {"author": author, "datetime": datetime, "raw_comment": comment, "clean_comment": preprocessing(comment),
                        "label": pred}
                
                print(record)
                
                record = json.dumps(record, ensure_ascii=False).encode("utf-8")        
                producer.send(topic='rawData', value=record)
                
            
    except KeyboardInterrupt:
        print('Stop flush!')
        pass


if __name__ == "__main__":
    main()