import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pymongo import MongoClient
#from pyspark.sql import Sparksession
#from pyspark.sql.functions import col
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


client = MongoClient('localhost',27017)
db = client['local']
#print(db)
data=db.amazon


#extracting specific cols from dataset
overall_list = list(data.find({}, {"overall":1,"_id": 0}, limit=3000))
overall_list = [x['overall'] for x in overall_list]
print(np.unique(overall_list, return_counts=True))

review_list = list(data.find({}, {"reviewTime":1,"_id": 0}, limit=3000))
review_list = [y['reviewTime'] for y in review_list]
print(np.unique(review_list, return_counts=True))

verified_list = list(data.find({}, {"verified":1,"_id": 0}, limit=3000))
verified_list = [z['verified'] for z in verified_list]






overall_list = list(data.find({}, {"overall":1,"_id": 0}, limit=3000))
overall_list = [x['overall'] for x in overall_list]

review_time_list = list(data.find({}, {"unixReviewTime":1,"_id": 0}, limit=3000))
review_time_list = [y['unixReviewTime'] for y in review_time_list]

corr = np.corrcoef(overall_list, review_time_list)[0,1]
print(f"Pearson Correlation coefficient between Overall rating and Unix Review Time is : {corr:.2f}")


verified_list = list(data.find({}, {"verified":1,"_id": 0}))
df_verified = pd.DataFrame(verified_list)
df_verified['verified'] = df_verified['verified'].replace({True: 'Verified', False: 'Not Verified'})
sns.countplot(x='verified', data=df_verified)
plt.xlabel('Verified Purchases')
plt.ylabel('Frequency')
plt.title('Distribution of Verified Purchases')
plt.show()


overall_list = list(data.find({}, {"overall":1,"_id": 0}))
df_overall = pd.DataFrame(overall_list)
sns.boxplot(x='overall', data=df_overall)
plt.xlabel('Overall Rating')
plt.title('Distribution of Overall Ratings')
plt.show()

review_list = list(data.find({}, {"reviewTime":1,"_id": 0}))
df_review = pd.DataFrame(review_list)
df_review['reviewTime'] = pd.to_datetime(df_review['reviewTime'], format='%m %d, %Y')
df_review = df_review.set_index('reviewTime').resample('M').size().reset_index(name='count')
sns.lineplot(x='reviewTime', y='count', data=df_review)
plt.xlabel('Review Time')
plt.ylabel('Review Frequency')
plt.title('Review Frequency Over Time')
plt.show()

df = pd.DataFrame(list(data.find({}, {"overall":1,"verified":1,"_id": 0})))
df['verified'] = df['verified'].replace({True: 'Verified', False: 'Not Verified'})
df = df.groupby(['overall', 'verified']).size().reset_index(name='count')
sns.catplot(x='overall', y='count', hue='verified', data=df, kind='bar')
plt.xlabel('Overall Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Overall Ratings by Verified Purchases')
plt.show()

df = pd.DataFrame(list(data.find({}, {"overall":1,"reviewTime":1,"verified":1,"_id": 0})))
df['verified'] = df['verified'].replace({True: 1, False: 0})
df['reviewTime'] = pd.to_datetime(df['reviewTime'], format='%m %d, %Y')
df['reviewTime'] = pd.to_numeric(df['reviewTime'])
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()



# histogram of overall col
plt.hist(overall_list, bins=np.arange(0.5, 6.6, 1),color = 'gray', edgecolor='black')
plt.xticks(np.arange(1, 6))
plt.xlabel('Overall Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Overall Ratings')
plt.show()



#histogram of review time col
plt.hist(review_list, bins=30 ,color = 'gray', edgecolor='black')
plt.xlabel('REVIEW TIME')
plt.ylabel('Frequency')
plt.title('Time of reviews')
plt.show()
