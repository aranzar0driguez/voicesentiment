#import whisper
import whisper_timestamped as whisper_ts
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, AutoModelForSequenceClassification
import numpy as np
from scipy.special import softmax
import nltk.data
from nltk.tokenize import RegexpTokenizer
from sentence import Sentence
import sentimentAnalysis


sentences = []

# Speech to text

audio = whisper_ts.load_audio("/Users/aranzarodriguez/Voice Sentiment Tracking/voicesentiment/client/audio/audio_test_2.m4a")
model = whisper_ts.load_model("tiny", device="cpu")
result = whisper_ts.transcribe(model, audio, language="en")


# Individual sentences...

nltk.download('punkt')
nltk.download('punkt_tab')
sentences = nltk.sent_tokenize(result['text'])

#Individual words... 

whispherTSWords = []

for segment in result["segments"]:
    for word in segment["words"]:
        
        averageTimeStamp = (word["start"] + word["end"])/2

        whispherTSWords.append({
            "word": word["text"],
            "timestamp": round(averageTimeStamp, 2)
        })



# Represents the index of the word we should start searching from in the timestampped words array 
pointer = 0
timestampedSentences = []

for sent in sentences:

    # Array of timestampped words the sentence contains 
    timestampWords = [] 

    # Takes aphostrophe's into consideration, though consider including hyphens or digits with alphnumerics...

    wordTokenizer = RegexpTokenizer(r"\b\w+(?:'\w+)?\b")
    tokenizedWords = wordTokenizer.tokenize(sent)

    # Looping through each of those tokenized words 
    for token in tokenizedWords:
        if token in whispherTSWords[pointer]["word"]:
            
            timestampWords.append(whispherTSWords[pointer])
            pointer+=1

    firstWordTime = timestampWords[0]['timestamp']
    lastWordTime = timestampWords[len(timestampWords) - 1]['timestamp']

    s = Sentence(sentence=sent, startTime=firstWordTime, endTime=lastWordTime)
    timestampedSentences.append(s)

# Sentiment Analysis

for sent in timestampedSentences:
    sent = sentimentAnalysis.analyzeSentiment(sent)
    print(sent.summary())
