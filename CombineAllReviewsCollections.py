import os
import glob
import pandas as pd
path="./FolderNameContainingAppReviewsCSVFiles"
New_file_name="Combined_CSV_TopFree_Trending_Reviews.csv"
os.chdir(path)
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
combined_csv.to_csv(New_file_name, index=False, encoding='utf-8-sig')
