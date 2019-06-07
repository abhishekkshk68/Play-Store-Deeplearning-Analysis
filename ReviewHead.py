import pandas as pd
import numpy as np
import os
path='./FolderContainingCombinedSheetOfAppsReviews/CombinedSheetOfReviewsDetails.csv'
data_from_Rev = pd.read_csv(path)

n_val=4472 #Unique app ID length
j=0
add_new_df=pd.DataFrame()
while(j<n_val):
          a=Value_Counts_list[0][j]
          if(j==0):
              add_new_df.append(df_test.query('rev_app_id == @a').head(20))
          else:
              add_new_df=add_new_df.append(df_test.query('rev_app_id == @a').head(20)) 
          if(j==4472):
             print('Final j val: ',j)
          j=j+1

add_new_df.reset_index(drop=True, inplace=True)            
print('shape after concat::',add_new_df.shape)
print('Done!')
