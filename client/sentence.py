class Sentence:
    #define the sentence object
  
    def __init__(self, sentence="", positive=0.0, negative=0.0, neutral=0.0, startTime=0.0, endTime=0.0):
        self.sentence = sentence
        self.positive = positive
        self.negative = negative 
        self.neutral = neutral
        self.startTime = startTime
        self.endTime = endTime

    def summary(self):
        return {
            "sentence": self.sentence,
            "timestamps" :{
                "start time": self.startTime,
                "end time": self.endTIme
            },
            "scores": {
                "postive": self.positive,
                "neutral": self.neutral,
                "negative": self.negative
            }
        }