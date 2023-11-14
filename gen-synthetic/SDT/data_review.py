import pickle
import json

data = pickle.load(open("/mlcv1/WorkingSpace/Personal/baotg/Kalapa/src/gen-synthetic/SDT/data/CASIA_ENGLISH/writer_dict.pkl", "rb"))
print(data)
for item in data:
    print(item)
# json.dump(data, open("/mlcv1/WorkingSpace/Personal/baotg/Kalapa/src/gen-synthetic/SDT/data/CASIA_ENGLISH/json/English_content.json", "w+"), indent=4, ensure_ascii=False )