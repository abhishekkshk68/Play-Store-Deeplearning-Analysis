import json
import unicodecsv as csv

import play_scraper

import pandas as pd

datafromcsv = pd.read_csv('./Scripts_details_response/TOP_FREE_WEATHER.csv')
App_Ids=datafromcsv.app_id

for n in range(len(App_Ids)):
	print(App_Ids[n])
ListOfReviews = play_scraper.reviews(App_Ids,10)
with open(".\\Scripts_details_response\\REV_TOP_FREE_WEATHER.json", "w") as f2:
  json.dump(ListOfReviews, f2)
f2.close()

df = pd.read_json('./Scripts_details_response/REV_TOP_FREE_WEATHER.json')
df.to_csv('./Scripts_details_response/REV_TOP_FREE_WEATHER.csv', encoding = 'utf-8')