import requests
import pickle
from pprint import pprint
from tqdm import tqdm


index = requests.get('https://cdn.jsdelivr.net/gh/thien0291/vietnam_dataset@1.0.0/Index.json').json()
result = {}

for k, v in tqdm(index.items()):
    code = v['code']
    result[k] = {}

    res = requests.get(f'https://cdn.jsdelivr.net/gh/thien0291/vietnam_dataset@1.0.0/data/{code}.json').json()
    for dis in res['district']:
        result[k][dis['name']] = []

        for street in dis['street']:
            if any(t in "0123456789" for t in street):
                continue

            if len(street) < 3 or len(street.split()) < 2:
                continue

            result[k][dis['name']].append(street)

pickle.dump(result, open('data/street.pkl', 'wb'))