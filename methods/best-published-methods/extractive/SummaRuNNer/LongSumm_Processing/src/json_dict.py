import json
import os

def clean(file):
    if 'import json' in file : return False
    file = file.split('{')
    if len(file) == 1: return False
    return '{' + '{'.join(file[1:])

if __name__ == '__main__':
    # files = os.listdir('json/')
    extractive_json_path = '../../datasets/talksumm/data/json-output/'
    files = os.listdir(extractive_json_path)

    data = []
    for ind,file in enumerate(files):
        with open(extractive_json_path+file,'r+') as f:
            text = clean(f.read())
            if text:
                text = json.loads(text)['metadata']
                if ind == 0: print(text.keys())
                # Change this line to change which sections are retained/dropped
                for key in text.keys() - ['title','sections','references','abstractText']:
                    text.pop(key)
                data.append(text)
            else:
                print("{} is bad".format(file))

    with open('../../replication/Summaformers/LongSumm_Processing/processed_data/listofdic_longsummtest_2.json','w+') as f:
        json.dump(data,f,indent=4)