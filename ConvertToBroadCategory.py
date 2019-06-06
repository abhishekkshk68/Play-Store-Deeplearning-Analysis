import pandas as pd
import numpy as np
#Categorical Column Encoding - Details Sheet 
data = pd.read_csv('./Combined_CSV_TopFree_Trending.csv')


def Num_Main_Cat(types):
    val=','
    abc = types
    fetched= ""
    if ((types =='[GAME]') & (types.find(val)== -1)):
        return 'GAMES'
    if ((types =='[GAME_ACTION]') & (types.find(val)== -1)):
        return 'GAMES'
    if ((types =='[GAME_ADVENTURE]')  & (types.find(val)== -1)):
        return 'GAMES'
    if ((types =='[GAME_ARCADE]')  & (types.find(val)== -1)):
        return 'GAMES'
    if ((types =='[GAME_BOARD]')  & (types.find(val)== -1)):
        return 'GAMES'
    if ((types =='[GAME_CARD]')  & (types.find(val)== -1)):
        return 'GAMES'
    if ((types =='[GAME_CASINO]')  & (types.find(val)== -1)):
        return 'GAMES'
    if ((types =='[GAME_CASUAL]')  & (types.find(val)== -1)):
        return 'GAMES'
    if ((types =='[GAME_EDUCATIONAL]')  & (types.find(val)== -1)):
        return 'GAMES'
    if ((types =='[GAME_MUSIC]')  & (types.find(val)== -1)):
        return 'GAMES'
    if ((types =='[GAME_PUZZLE]')  & (types.find(val)== -1)):
        return 'GAMES'
    if ((types =='[GAME_RACING]')  & (types.find(val)== -1)):
        return 'GAMES'
    if ((types =='[GAME_ROLE_PLAYING]')  & (types.find(val)== -1)):
        return 'GAMES'
    if ((types =='[GAME_SIMULATION]')  & (types.find(val)== -1)):
        return 'GAMES'
    if ((types =='[GAME_SPORTS]')  & (types.find(val)== -1)):
        return 'GAMES'
    if ((types =='[GAME_STRATEGY]')  & (types.find(val)== -1)):
        return 'GAMES'
    if ((types =='[GAME_TRIVIA]')  & (types.find(val)== -1)):
        return 'GAMES'
    if ((types =='[GAME_WORD]')  & (types.find(val)== -1)):
        return 'GAMES'
    if ((types =='[GAME_WORD]')  & (types.find(val)== -1)):
        return 'GAMES'
    if ((types =='[FAMILY]')  & (types.find(val)== -1)):
        return 'FAMILY_ORIENTED'
    if ((types =='[FAMILY_ACTION]')  & (types.find(val)== -1)):
        return 'FAMILY_ORIENTED'
    if ((types =='[FAMILY_BRAINGAMES]')  & (types.find(val)== -1)):
        return 'FAMILY_ORIENTED'
    if ((types =='[FAMILY_CREATE]')  & (types.find(val)== -1)):
        return 'FAMILY_ORIENTED'
    if ((types =='[FAMILY_EDUCATION]')  & (types.find(val)== -1)):
        return 'FAMILY_ORIENTED'
    if ((types =='[FAMILY_MUSICVIDEO]')  & (types.find(val)== -1)):
        return 'FAMILY_ORIENTED'
    if ((types =='[FAMILY_PRETEND]')  & (types.find(val)== -1)):
        return 'FAMILY_ORIENTED'
    if ((types =='[FINANCE]')  & (types.find(val)== -1)):
        return 'FINANCE'
    if ((types =='[MEDICAL]')  & (types.find(val)== -1)):
        return 'HEALTHCARE'
    if ((types =='[HEALTH_AND_FITNESS]')  & (types.find(val)== -1)):
        return 'HEALTHCARE'
    if ((types =='[SPORTS]')  & (types.find(val)== -1)):
        return 'GAMES'
    if ((types =='[MAPS_AND_NAVIGATION]')  & (types.find(val)== -1)):
        return 'MAPS_AND_TRAVEL'
    if ((types =='[TRAVEL_AND_LOCAL]')  & (types.find(val)== -1)):
        return 'MAPS_AND_TRAVEL'
    if ((types =='[HOUSE_AND_HOME]')  & (types.find(val)== -1)):
        return 'HOUSING'
    if ((types =='[LIFESTYLE]')  & (types.find(val)== -1)):
        return 'PERFORMANCE_BOOST'
    if ((types =='[BEAUTY]')  & (types.find(val)== -1)):
        return 'PERFORMANCE_BOOST'
    if ((types =='[PARENTING]')  & (types.find(val)== -1)):
        return 'PERFORMANCE_BOOST'
    if ((types =='[PERSONALIZATION]')  & (types.find(val)== -1)):
        return 'PERFORMANCE_BOOST'
    if ((types =='[PRODUCTIVITY]')  & (types.find(val)== -1)):
        return 'PERFORMANCE_BOOST'
    if ((types =='[TOOLS]')  & (types.find(val)== -1)):
        return 'PERFORMANCE_BOOST'
    if ((types =='[VIDEO_PLAYERS]')  & (types.find(val)== -1)):
        return 'PERFORMANCE_BOOST'
    if ((types =='[ENTERTAINMENT]')  & (types.find(val)== -1)):
        return 'ENTERTAINMENT_LIST'
    if ((types =='[MUSIC_AND_AUDIO]')  & (types.find(val)== -1)):
        return 'ENTERTAINMENT_LIST'
    if ((types =='[COMICS]')  & (types.find(val)== -1)):
        return 'ENTERTAINMENT_LIST'
    if ((types =='[PHOTOGRAPHY]')  & (types.find(val)== -1)):
        return 'ENTERTAINMENT_LIST'
    if ((types =='[EVENTS]')  & (types.find(val)== -1)):
        return 'ENTERTAINMENT_LIST'
    if ((types =='[SHOPPING]')  & (types.find(val)== -1)):
        return 'ENTERTAINMENT_LIST'
    if ((types =='[FOOD_AND_DRINK]')  & (types.find(val)== -1)):
        return 'FOOD_AND_DRINK'
    if ((types =='[ART_AND_DESIGN]')  & (types.find(val)== -1)):
        return 'ART_AND_DESIGN'
    if ((types =='[AUTO_AND_VEHICLES]')  & (types.find(val)== -1)):
        return 'AUTOSHOP'
    if ((types =='[BOOKS_AND_REFERENCE]')  & (types.find(val)== -1)):
        return 'KNOWLEDGE'
    if ((types =='[BUSINESS]')  & (types.find(val)== -1)):
        return 'KNOWLEDGE'
    if ((types =='[EDUCATION]')  & (types.find(val)== -1)):
        return 'KNOWLEDGE'
    if ((types =='[LIBRARIES_AND_DEMO]')  & (types.find(val)== -1)):
        return 'KNOWLEDGE'
    if ((types =='[NEWS_AND_MAGAZINES]')  & (types.find(val)== -1)):
        return 'NEWS'
    if ((types =='[COMMUNICATION]')  & (types.find(val)== -1)):
        return 'COMMUNICATION'
    if ((types =='[DATING]')  & (types.find(val)== -1)):
        return 'COMMUNICATION'
    if ((types =='[SOCIAL]')  & (types.find(val)== -1)):
        return 'COMMUNICATION'
    if ((types =='[WEATHER]')  & (types.find(val)== -1)):
        return 'WEATHER'
    if ((types =='[ANDROID_WEAR]')  & (types.find(val)== -1)):
        return 'PERFORMANCE_BOOST'
    if (abc.find(val)!= -1):
        fetched = abc[1:(abc.index(','))]
        abc=""
        if (fetched =='GAME'):
            return 'GAMES'
        if (fetched =='GAME_ACTION'):
            return 'GAMES'
        if (fetched =='GAME_ADVENTURE'):
            return 'GAMES'
        if (fetched =='GAME_ARCADE'):
            return 'GAMES'
        if (fetched =='GAME_BOARD'):
            return 'GAMES'
        if (fetched =='GAME_CARD'):
            return 'GAMES'
        if (fetched =='GAME_CASINO'):
            return 'GAMES'
        if (fetched =='GAME_CASUAL'):
            return 'GAMES'
        if (fetched =='GAME_EDUCATIONAL'):
            return 'GAMES'
        if (fetched =='GAME_MUSIC'):
            return 'GAMES'
        if (fetched =='GAME_PUZZLE'):
            return 'GAMES'
        if (fetched =='GAME_RACING'):
            return 'GAMES'
        if (fetched =='GAME_ROLE_PLAYING'):
            return 'GAMES'
        if (fetched =='GAME_SIMULATION'):
            return 'GAMES'
        if (fetched =='GAME_SPORTS'):
            return 'GAMES'
        if (fetched =='GAME_STRATEGY'):
            return 'GAMES'
        if (fetched =='GAME_TRIVIA'):
            return 'GAMES'
        if (fetched =='GAME_WORD'):
            return 'GAMES'
        if (fetched =='GAME_WORD'):
            return 'GAMES'
        if (fetched =='FAMILY'):
            return 'FAMILY_ORIENTED'
        if (fetched =='FAMILY_ACTION'):
            return 'FAMILY_ORIENTED'
        if (fetched =='FAMILY_BRAINGAMES'):
            return 'FAMILY_ORIENTED'
        if (fetched =='FAMILY_CREATE'):
            return 'FAMILY_ORIENTED'
        if (fetched =='FAMILY_EDUCATION'):
            return 'FAMILY_ORIENTED'
        if (fetched =='FAMILY_MUSICVIDEO'):
            return 'FAMILY_ORIENTED'
        if (fetched =='FAMILY_PRETEND'):
            return 'FAMILY_ORIENTED'
        if (fetched =='FINANCE'):
            return 'FINANCE'
        if (fetched =='MEDICAL'):
            return 'HEALTHCARE'
        if (fetched =='HEALTH_AND_FITNESS'):
            return 'HEALTHCARE'
        if (fetched =='SPORTS'):
            return 'GAMES'
        if (fetched =='MAPS_AND_NAVIGATION'):
            return 'MAPS_AND_TRAVEL'
        if (fetched =='TRAVEL_AND_LOCAL'):
            return 'MAPS_AND_TRAVEL'
        if (fetched =='HOUSE_AND_HOME'):
            return 'HOUSING'
        if (fetched =='LIFESTYLE'):
            return 'PERFORMANCE_BOOST'
        if (fetched =='BEAUTY'):
            return 'PERFORMANCE_BOOST'
        if (fetched =='PARENTING'):
            return 'PERFORMANCE_BOOST'
        if (fetched =='PERSONALIZATION'):
            return 'PERFORMANCE_BOOST'
        if (fetched =='PRODUCTIVITY'):
            return 'PERFORMANCE_BOOST'
        if (fetched =='TOOLS'):
            return 'PERFORMANCE_BOOST'
        if (fetched =='VIDEO_PLAYERS'):
            return 'PERFORMANCE_BOOST'
        if (fetched =='ENTERTAINMENT'):
            return 'ENTERTAINMENT_LIST'
        if (fetched =='MUSIC_AND_AUDIO'):
            return 'ENTERTAINMENT_LIST'
        if (fetched =='COMICS'):
            return 'ENTERTAINMENT_LIST'
        if (fetched =='PHOTOGRAPHY'):
            return 'ENTERTAINMENT_LIST'
        if (fetched =='EVENTS'):
            return 'ENTERTAINMENT_LIST'
        if (fetched =='SHOPPING'):
            return 'ENTERTAINMENT_LIST'
        if (fetched =='FOOD_AND_DRINK'):
            return 'FOOD_AND_DRINK'
        if (fetched =='ART_AND_DESIGN'):
            return 'ART_AND_DESIGN'
        if (fetched =='AUTO_AND_VEHICLES'):
            return 'AUTOSHOP'
        if (fetched =='BOOKS_AND_REFERENCE'):
            return 'KNOWLEDGE'
        if (fetched =='BUSINESS'):
            return 'KNOWLEDGE'
        if (fetched =='EDUCATION'):
            return 'KNOWLEDGE'
        if (fetched =='LIBRARIES_AND_DEMO'):
            return 'KNOWLEDGE'
        if (fetched =='NEWS_AND_MAGAZINES'):
            return 'NEWS'
        if (fetched =='COMMUNICATION'):
            return 'COMMUNICATION'
        if (fetched =='DATING'):
            return 'COMMUNICATION'
        if (fetched =='SOCIAL'):
            return 'COMMUNICATION'
        if (fetched =='WEATHER'):
            return 'WEATHER'
        if (fetched =='ANDROID_WEAR'):
            return 'PERFORMANCE_BOOST'
data['Main_Category'] = data['category'].map(Num_Main_Cat)
data.to_csv('./Combined_CSV_TopFree_Trending_Updated.csv',encoding='utf-8')