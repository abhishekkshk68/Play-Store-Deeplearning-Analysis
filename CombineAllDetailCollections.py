import os
import glob
import pandas as pd
os.chdir("./Scripts_details_response/FinalDataSet_AndroidApps/Combined_data/Trial")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
combined_csv.to_csv( "Combined_CSV_TopFree_Trending.csv", index=False, encoding='utf-8-sig')