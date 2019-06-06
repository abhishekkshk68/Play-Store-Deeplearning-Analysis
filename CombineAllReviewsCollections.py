import os
import glob
import pandas as pd
os.chdir("./Scripts_details_response/FinalDataSet_AndroidApps/Combined_Reviews")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
combined_csv.to_csv("./Scripts_details_response/FinalDataSet_AndroidApps/CombinedReviews_TopFree_Trending.csv", index=False, encoding='utf-8-sig')
