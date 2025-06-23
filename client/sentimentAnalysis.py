
import csv
import urllib.request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
from sentence import Sentence

def analyzeSentiment(sent):

        task='sentiment'
        MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)

        #Labeling for sentiment (neg, pos, neu)
        labels=[]
        mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
        with urllib.request.urlopen(mapping_link) as f:
            html = f.read().decode('utf-8').split("\n")
            csvreader = csv.reader(html, delimiter='\t')
        labels = [row[1] for row in csvreader if len(row) > 1]

        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        tokenizer.save_pretrained(MODEL)
        model.save_pretrained(MODEL)
        encoded_input = tokenizer(sent.sentence, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores) # transform's the model's output into a # between 0 -1 

        sentimentScores = {}

        for i in range(scores.shape[0]):

            sentimentScores[labels[i]] = scores[i]

        sent.negative = sentimentScores['negative']
        sent.positive = sentimentScores['positive']
        sent.neutral = sentimentScores['neutral']
        
        return sent
