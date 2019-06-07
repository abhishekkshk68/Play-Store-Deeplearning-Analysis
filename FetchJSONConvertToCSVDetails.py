import json
import unicodecsv as csv

import play_scraper

import pandas as pd

collection_name='TRENDING'
category_name'ENTERTAINMENT'
path_json ='./FolderNameToSaveResponseFiles/ENTERTAINMENT.json'
path_csv ='./FolderNameToSaveResponseFiles/ENTERTAINMENT.csv'

get_data = (play_scraper.collection(collection = collection_name, category = , results = 120, page = 1, detailed = True))
with open(".\\FolderNameToSaveResponseFiles\\ENTERTAINMENT.json", "w") as f1:
  json.dump(get_data, f1)
f1.close()

df = pd.read_json(path_json)
df.to_csv(path_csv, encoding = 'utf-8')
