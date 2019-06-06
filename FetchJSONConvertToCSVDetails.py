import json
import unicodecsv as csv

import play_scraper

import pandas as pd

get_data = (play_scraper.collection(collection = 'TRENDING', category = 'ENTERTAINMENT', results = 120, page = 1, detailed = True))
with open(".\\Scripts_details_response\\ENTERTAINMENT.json", "w") as f1:
  json.dump(get_data, f1)
f1.close()

df = pd.read_json('./Scripts_details_response/ENTERTAINMENT.json')
df.to_csv('./Scripts_details_response/ENTERTAINMENT.csv', encoding = 'utf-8')