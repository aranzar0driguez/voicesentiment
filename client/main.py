#import whisper
import whisper_timestamped as whisper_ts
import json
import matplotlib

audio = whisper_ts.load_audio("/Users/aranzarodriguez/Voice Sentiment Tracking/voicesentiment/client/audio/audio_test_1.m4a")
model = whisper_ts.load_model("tiny", device="cpu")
result = whisper_ts.transcribe(model, audio, language="en")

print(f"Recording text: \n{result['text']}")

wordsMapping = {}

for segment in result["segments"]:
    for word in segment["words"]:
        
        averageTimeStamp = (word["start"] + word["end"])/2
        wordsMapping[word["text"]] = round(averageTimeStamp, 2)

print(wordsMapping)
        




#print(json.dumps(result, indent=2, ensure_ascii = False))