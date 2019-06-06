#Combine Total Average Rating , Installs , Total Number of Reviews , Required Android Version , Main Category Unique Val as per Application ID.


#Required Android Version
ReqAndroidVer_Apps=data_from_CleanedDetails[['required_android_version','app_id']]

ReqAndroidVer_Apps_New=ReqAndroidVer_Apps
ReqAndroidVer_Apps_List=ReqAndroidVer_Apps_New.values.T.tolist()
print("**************************************************")   
print('ReqAndroidVer_Apps_List length ::',len(ReqAndroidVer_Apps_List))
DetailsLen=int(7762)
def HelpReqAndroidVer(z_app):
  j=0
  i=1
  reqandroid_ver=0
  while j < (DetailsLen):
      if((ReqAndroidVer_Apps_List[i][j])==z_app):
            reqandroid_ver=ReqAndroidVer_Apps_List[i-1][j]
            #print('reqandroid_ver:',reqandroid_ver)
            break
      elif(((ReqAndroidVer_Apps_List[i][j])!=z_app) & (j<DetailsLen)):
            j=j+1
            continue
  return reqandroid_ver    
data_from_Rev['required_android_version'] = data_from_Rev['rev_app_id'].apply(HelpReqAndroidVer)


#Append Broad Categories From Details Sheet To Reviews Sheet .



#data_from_CleanedReviews :: Dataframe consisting of Cleaned Reviews Sheet 
#data_from_CleanedDetails  ::  Dataframe consisting of Cleaned Details Sheet 
##Reading From Cleaned Details sheet and combining important columns to Cleaned Reviews Sheet.
fetched_0=data_from_CleanedDetails.loc[data_from_CleanedDetails['Category_UniqueVal'] == 0]
App_id_Cat0=fetched_0.app_id
List_App_id_Cat0=App_id_Cat0.unique()
List_App_id_Cat0_cleaned = [item.replace(' ',',') for item in List_App_id_Cat0]

fetched_1=data_from_CleanedDetails.loc[data_from_CleanedDetails['Category_UniqueVal'] == 1]
List_App_id_Cat1=fetched_1.app_id.unique()
List_App_id_Cat1_cleaned = [item.replace(' ',',') for item in List_App_id_Cat1]

fetched_2=data_from_CleanedDetails.loc[data_from_CleanedDetails['Category_UniqueVal'] == 2]
List_App_id_Cat2=fetched_2.app_id.unique()
List_App_id_Cat2_cleaned = [item.replace(' ',',') for item in List_App_id_Cat2]

fetched_3=data_from_CleanedDetails.loc[data_from_CleanedDetails['Category_UniqueVal'] == 3]
List_App_id_Cat3=fetched_3.app_id.unique()
List_App_id_Cat3_cleaned = [item.replace(' ',',') for item in List_App_id_Cat3]

fetched_4=data_from_CleanedDetails.loc[data_from_CleanedDetails['Category_UniqueVal'] == 4]
List_App_id_Cat4=fetched_4.app_id.unique()
List_App_id_Cat4_cleaned = [item.replace(' ',',') for item in List_App_id_Cat4]

fetched_5=data_from_CleanedDetails.loc[data_from_CleanedDetails['Category_UniqueVal'] == 5]
List_App_id_Cat5=fetched_5.app_id.unique()
List_App_id_Cat5_cleaned = [item.replace(' ',',') for item in List_App_id_Cat5]

fetched_6=data_from_CleanedDetails.loc[data_from_CleanedDetails['Category_UniqueVal'] == 6]
List_App_id_Cat6=fetched_6.app_id.unique()
List_App_id_Cat6_cleaned = [item.replace(' ',',') for item in List_App_id_Cat6]

fetched_7=data_from_CleanedDetails.loc[data_from_CleanedDetails['Category_UniqueVal'] == 7]
List_App_id_Cat7=fetched_7.app_id.unique()
List_App_id_Cat7_cleaned = [item.replace(' ',',') for item in List_App_id_Cat7]

fetched_8=data_from_CleanedDetails.loc[data_from_CleanedDetails['Category_UniqueVal'] == 8]
List_App_id_Cat8=fetched_8.app_id.unique()
List_App_id_Cat8_cleaned = [item.replace(' ',',') for item in List_App_id_Cat8]

fetched_9=data_from_CleanedDetails.loc[data_from_CleanedDetails['Category_UniqueVal'] == 9]
List_App_id_Cat9=fetched_9.app_id.unique()
List_App_id_Cat9_cleaned = [item.replace(' ',',') for item in List_App_id_Cat9]

fetched_10=data_from_CleanedDetails.loc[data_from_CleanedDetails['Category_UniqueVal'] == 10]
List_App_id_Cat10=fetched_10.app_id.unique()
List_App_id_Cat10_cleaned = [item.replace(' ',',') for item in List_App_id_Cat10]

fetched_11=data_from_CleanedDetails.loc[data_from_CleanedDetails['Category_UniqueVal'] == 11]
List_App_id_Cat11=fetched_11.app_id.unique()
List_App_id_Cat11_cleaned = [item.replace(' ',',') for item in List_App_id_Cat11]

fetched_12=data_from_CleanedDetails.loc[data_from_CleanedDetails['Category_UniqueVal'] == 12]
List_App_id_Cat12=fetched_12.app_id.unique()
List_App_id_Cat12_cleaned = [item.replace(' ',',') for item in List_App_id_Cat12]

#Adding Main_Category_UniqueVal column 
def CheckCat(App_Ids_Reviews):
    if App_Ids_Reviews in List_App_id_Cat0_cleaned:
        return 0
    elif App_Ids_Reviews in List_App_id_Cat1_cleaned:
        return 1
    elif App_Ids_Reviews in List_App_id_Cat2_cleaned:
        return 2
    elif App_Ids_Reviews in List_App_id_Cat3_cleaned:
        return 3
    elif App_Ids_Reviews in List_App_id_Cat4_cleaned:
        return 4
    elif App_Ids_Reviews in List_App_id_Cat5_cleaned:
        return 5
    elif App_Ids_Reviews in List_App_id_Cat6_cleaned:
        return 6
    elif App_Ids_Reviews in List_App_id_Cat7_cleaned:
        return 7
    elif App_Ids_Reviews in List_App_id_Cat8_cleaned:
        return 8
    elif App_Ids_Reviews in List_App_id_Cat9_cleaned:
        return 9
    elif App_Ids_Reviews in List_App_id_Cat10_cleaned:
        return 10
    elif App_Ids_Reviews in List_App_id_Cat11_cleaned:
        return 11
    elif App_Ids_Reviews in List_App_id_Cat12_cleaned:
        return 12
    else:
        return 20
    
data_from_CleanedReviews['Main_Category_UniqueVal'] = data_from_CleanedReviews['rev_app_id'].map(CheckCat)

print("Added unique Category to Reviews sheet")


#Average Rate and Review Count of Apps
ReviewCount_Ratings_Apps=data_from_CleanedDetails[['reviews','app_id','score']]


ReviewCount_Ratings_Apps_1=ReviewCount_Ratings_Apps
RevRatAppList=ReviewCount_Ratings_Apps_1.values.T.tolist()
print('RevRatAppList length ::',len(RevRatAppList))
DetailsLen=int(10404)
def HelpReviews(z_app):
  j=0
  i=1
  while j < (DetailsLen):
      if((RevRatAppList[i][j])==z_app):           
            revno=RevRatAppList[i-1][j]
            break
      elif(((RevRatAppList[i][j])!=z_app) & (j<DetailsLen)):
            j=j+1
            continue
      elif(((RevRatAppList[i][j])!=z_app) & (j==DetailsLen-1)):
            revno=0
            break            
  return revno    
Implementation_df_1['TotalNumOfReviews'] = Implementation_df_1['rev_app_id'].apply(HelpReviews)


print('RevRatAppList length ::',len(RevRatAppList))
DetailsLen=int(10404)

def HelpRatings(z_app):
  j=0
  i=1
  while j < (DetailsLen):
      if((RevRatAppList[i][j])==z_app ):            
            ratingno=RevRatAppList[i+1][j]
            break
      elif(((RevRatAppList[i][j])!=z_app) & (j<DetailsLen)):
            j=j+1
            continue
      elif(((RevRatAppList[i][j])!=z_app) & (j==DetailsLen-1)):
            ratingno=0
            break            
  return ratingno    
Implementation_df_1['TotalAverageRating'] = Implementation_df_1['rev_app_id'].apply(HelpRatings)



