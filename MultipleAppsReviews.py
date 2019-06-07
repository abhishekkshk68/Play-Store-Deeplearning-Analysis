import json
import unicodecsv as csv
import play_scraper
import pandas as pd
path_appdetails='./FolderContainingAppDetailsCSV/TOP_FREE_WEATHER.csv'
path_appreviews_json='./FolderToAddAppReviews/REV_TOP_FREE_WEATHER.json'
path_appreviews_csv='./FolderToAddAppReviews/REV_TOP_FREE_WEATHER.csv'

datafromcsv = pd.read_csv(path_appdetails)
App_Ids=datafromcsv.app_id

for n in range(len(App_Ids)):
	print(App_Ids[n])
ListOfReviews = play_scraper.reviews(App_Ids,10)
with open(".\\FolderToAddAppReviews\\REV_TOP_FREE_WEATHER.json", "w") as f2:
  json.dump(ListOfReviews, f2)
f2.close()

df = pd.read_json(path_appreviews_json)
df.to_csv(path_appreviews_csv, encoding = 'utf-8')
