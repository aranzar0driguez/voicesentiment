import whisper_timestamped as whisper_ts
from scipy.special import softmax
import nltk.data
from nltk.tokenize import RegexpTokenizer
from sentence import Sentence
import sentimentAnalysis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import plotly.express as px

sentences = []

# Speech to text

audio = whisper_ts.load_audio("/Users/aranzarodriguez/Voice Sentiment Tracking/voicesentiment/client/audio/audio_test_3.m4a")
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

data = {
    "sentence": [],
    "positive": [],
    "negative": [],
    "timestamp": []
}

for sent in timestampedSentences:
    
    sent = sentimentAnalysis.analyzeSentiment(sent)
    #print(sent.summary())

    data["sentence"].append(sent.sentence)
    data["positive"].append(sent.positive)
    data["negative"].append(sent.negative)
    data["timestamp"].append((sent.startTime + sent.endTime)/2)

df = pd.DataFrame(data)
print(df.head())

# Re-structure the data frame... 
df_melted = pd.melt(
    df, 
    id_vars=["timestamp", "sentence"], # Fixed columns 
    value_vars=["positive", "negative"], # columns that will be merged together 
    var_name="sentiment", # new name of the column that will hold positive/negative together
    value_name="score") # name of the column that will hold the old value that used to be under positive/negative 

print(df_melted)

# Visual distribution of sentiment 

# Create interactive plot
fig = px.line(
    df_melted,
    x="timestamp",
    y="score",
    color="sentiment",
    markers=True,
    line_shape="spline",
    hover_data=["sentence"],
    color_discrete_map={"positive": "green", "negative": "red"}
)

fig.update_layout(title="Sentiment Scores Over Time", yaxis_range=[0,1])

fig.show()

'''
sns.set_theme(style="whitegrid")

sns.lineplot(data=df_melted, x="timestamp", y="score", hue="sentiment", palette={"positive": "green", "negative": "red"})


plt.title("Negative Sentiment Score Over Time")
plt.ylim(0,1)
plt.xlabel("Time (s)")
plt.ylabel("Negative Sentiment Score")
plt.show()
'''

