# Play-Store-Deeplearning-Analysis

## Work Flow of Web Scraping:

This play scraper is a python API. The following steps indicate the stage-by-stage function of
each script used for getting the data from the play store.

• FetchJSONConvertToCSVDetails.py :
This script takes following parameters
1. collection: Specify the name of the collection as TRENDING or TOPFREE.
2. category : Give one category name from the list of 58 categories mentioned in the table 2.
3. path : The user can specify their own path else by default, the script would take the current
directory to create a folder and save the response files.
WorkFlow :
The collection method of the play scraper is invoked by passing these parameters. The utils.py
file in the play scraper package collects the HTML response returned from the request and
constructs a JSON from it. This JSON data is received in response and a copy is stored in this
directory. Using pandas, we convert this JSON structure to CSV file format.

• MultipleAppsReviews.py :
This script takes just one parameter:
1. path: The CSV file path of app details belonging to one category is given.
WorkFlow :
This script reads application ID column from the given sheet and creates a list. This list is further
passed on as a parameter while invoking the reviews method of play scraper.

• CombineAllDetailCollections.py :
This script too takes two parameters:
1. path: The file path of all the 58 categories of sheets under each collection, Top Free and
Trending details, is kept under the same path. The same is mentioned here.
2. New File Name: After merging the details contained in all the categories into a new sheet, a
new filename is assigned to it. The user can specify the name in this parameter.
WorkFlow :
The concat function provided by pandas is used to combine the data read from all the sheets
falling under the specified path.

• CombineAllReviewsCollections.py :
This script too takes two parameters:
1. path: The file path of all the 58 categories of sheets under each collection, Top Free and
Trending applications’ reviews, is kept under the same path. The same is mentioned here.
2. New File Name: After merging the reviews contained in all the categories into a new sheet, a
new filename is assigned to it. The user can specify the name in this parameter.
WorkFlow :
The concat function provided by pandas is used to combine the data read from all the sheets
falling under the specified path.

• ConvertToBroadCategory.py :
This script too takes two parameters :
1. Original File Name: This parameter specifies the entire path and file name to be processed.
2. New File Name : The updated file is saved with a new name specified by this parameter.
WorkFlow :
One application can be under multiple categories too. So the combined sheet consisted of a
category column that involved multiple categories seperated by comma. We segregate the
categories into broad sections by using this script.

• CombineDetailsToReviews.py :
This script does not take any parameter.

WorkFlow :
After narrowing down to the required columns which can be used for processing in the details
sheet, those collective set of columns are combined to the reviews sheet. 
If the user wishes to continue using the same specified set of columns as used in this study, for data analysis, they can execute this script. 
Final Set of Columns that was combined to the Reviews Sheet as per Application ID: 
Total Average Rating, Installs, Total Number of Reviews, Required Android Version, Main Category Unique Val.

• ReviewHead.py :
This script takes one parameter:
1. File Path: This parameter specifies the entire path and file name to be processed for balancing
the dataset.
WorkFlow :
The combined dataset consisted of varying number of rows for each application. To balance out
the data frame, this script collects the top 20 rows of reviews of each application id.
