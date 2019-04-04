
# coding: utf-8

# ## Create datasets for the Content-based Filter

# This notebook builds the data we will use for creating our content based model. We'll collect the data via a collection of SQL queries from the publicly avialable Kurier.at dataset in BigQuery.
# Kurier.at is an Austrian newsite. The goal of these labs is to recommend an article for a visitor to the site. In this lab we collect the data for training. In the subsequent notebook we train the recommender model. 
# 
# This notebook illustrates
# * how to pull data from BigQuery table and write to local files
# * how to make reproducible train and test splits 

# In[1]:


import os
import tensorflow as tf
import numpy as np
import google.datalab.bigquery as bq

output = os.popen("gcloud config get-value project").readlines()
project_name = output[0][:-1]

# change these to try this notebook out
PROJECT = project_name
BUCKET = project_name
#BUCKET = BUCKET.replace("qwiklabs-gcp-", "inna-bckt-")
REGION = 'europe-west1'  ## note: Cloud ML Engine not availabe in europe-west3!


# do not change these
os.environ['PROJECT'] = PROJECT
os.environ['BUCKET'] = BUCKET
os.environ['REGION'] = REGION
os.environ['TFVERSION'] = '1.8'

print(PROJECT)
print(BUCKET)
print(REGION)


# In[2]:


get_ipython().run_cell_magic('bash', '', 'gcloud  config  set project $PROJECT\ngcloud config set compute/region $REGION')


# We will use this helper funciton to write lists containing article ids, categories, and authors for each article in our database to local file. 

# In[4]:


def write_list_to_disk(my_list, filename):
  with open(filename, 'w') as f:
    for item in my_list:
        line = "%s\n" % item
        f.write(str(line.encode('utf8')))


# ### Pull data from BigQuery
# 
# The cell below creates a local text file containing all the article ids (i.e. 'content ids') in the dataset. 
# 
# Have a look at the original dataset in [BigQuery](https://bigquery.cloud.google.com/welcome/). Then read through the query below and make sure you understand what it is doing. 

# In[5]:


sql="""
#standardSQL
SELECT  
  (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(hits.customDimensions)) AS content_id 
FROM `cloud-training-demos.GA360_test.ga_sessions_sample`,   
  UNNEST(hits) AS hits
WHERE 
  # only include hits on pages
  hits.type = "PAGE"
  AND (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(hits.customDimensions)) IS NOT NULL
GROUP BY
  content_id
"""
content_ids_list = bq.Query(sql).execute().result().to_dataframe()['content_id'].tolist()
write_list_to_disk(content_ids_list, "content_ids.txt")
print("Some sample content IDs {}".format(content_ids_list[:3]))
print("The total number of articles is {}".format(len(content_ids_list)))


# In[19]:


sql="""
#standardSQL
SELECT  *
FROM `cloud-training-demos.GA360_test.ga_sessions_sample`
LIMIT 10
"""
bq.Query(sql).execute().result()


# In the following cells, you will create a local file which contains a list of article categories and a list of article authors.
# 
# **Hint**: For the TODO below, modify the query above changing 'content_id' to the necessary field and changing index=10 
# to the appropriate index.  
# Refer back to the original dataset, use the `hits.customDimensions.index` 
# field to find the correct index.

# In[21]:


TODO="""
TODO: Modify the query above to instead create a list of all categories in the dataset.
You'll need to change the content_id to the appropriate field as well as the index. 
"""

sql="""
#standardSQL
SELECT  
  (SELECT MAX(IF(index=7, value, NULL)) FROM UNNEST(hits.customDimensions)) AS category 
FROM `cloud-training-demos.GA360_test.ga_sessions_sample`,   
  UNNEST(hits) AS hits
WHERE 
  # only include hits on pages
  hits.type = "PAGE"
  AND (SELECT MAX(IF(index=7, value, NULL)) FROM UNNEST(hits.customDimensions)) IS NOT NULL
GROUP BY
  category
"""
#TODO: Modify the query above to create the list of categories
categories_list = bq.Query(sql).execute().result().to_dataframe()['category'].tolist() 
write_list_to_disk(categories_list, "categories.txt")
print(categories_list)


# When creating the author list, we'll only use the first author information for each article.  

# In[22]:


sql="""
#standardSQL
SELECT
  REGEXP_EXTRACT((SELECT MAX(IF(index=2, value, NULL)) FROM UNNEST(hits.customDimensions)), r"^[^,]+")  AS first_author  
FROM `cloud-training-demos.GA360_test.ga_sessions_sample`,   
  UNNEST(hits) AS hits
WHERE 
  # only include hits on pages
  hits.type = "PAGE"
  AND (SELECT MAX(IF(index=2, value, NULL)) FROM UNNEST(hits.customDimensions)) IS NOT NULL
GROUP BY   
  first_author
"""
authors_list = bq.Query(sql).execute().result().to_dataframe()['first_author'].tolist()
write_list_to_disk(authors_list, "authors.txt")
print("Some sample authors {}".format(authors_list[:10]))
print("The total number of authors is {}".format(len(authors_list)))


# ### Create train and test sets.
# 
# In this section, we will create the train/test split of our data for training our model. Read through the query and complete the TODO at the bottom.  
# Use the concatenated values for visitor id and content id to create a farm fingerprint, taking 90% of the data for the training set. 

# In[25]:


sql="""
WITH site_history as (
  SELECT
      fullVisitorId as visitor_id,
      (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(hits.customDimensions)) AS content_id,
      (SELECT MAX(IF(index=7, value, NULL)) FROM UNNEST(hits.customDimensions)) AS category, 
      (SELECT MAX(IF(index=6, value, NULL)) FROM UNNEST(hits.customDimensions)) AS title,
      (SELECT MAX(IF(index=2, value, NULL)) FROM UNNEST(hits.customDimensions)) AS author_list,
      SPLIT(RPAD((SELECT MAX(IF(index=4, value, NULL)) FROM UNNEST(hits.customDimensions)), 7), '.') as year_month_array,
      LEAD(hits.customDimensions, 1) OVER (PARTITION BY fullVisitorId ORDER BY hits.time ASC) as nextCustomDimensions
  FROM 
    `cloud-training-demos.GA360_test.ga_sessions_sample`,   
     UNNEST(hits) AS hits
   WHERE 
     # only include hits on pages
      hits.type = "PAGE"
      AND
      fullVisitorId IS NOT NULL
      AND
      hits.time != 0
      AND
      hits.time IS NOT NULL
      AND
      (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(hits.customDimensions)) IS NOT NULL
)
SELECT
  visitor_id,
  content_id,
  category,
  REGEXP_REPLACE(title, r",", "") as title,
  REGEXP_EXTRACT(author_list, r"^[^,]+") as author,
  DATE_DIFF(DATE(CAST(year_month_array[OFFSET(0)] AS INT64), CAST(year_month_array[OFFSET(1)] AS INT64), 1), DATE(1970,1,1), MONTH) as months_since_epoch,
  (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(nextCustomDimensions)) as next_content_id
FROM
  site_history
WHERE (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(nextCustomDimensions)) IS NOT NULL
      AND MOD(ABS(FARM_FINGERPRINT(CONCAT(visitor_id, content_id))), 10) < 9
"""
training_set_df = bq.Query(sql).execute().result().to_dataframe()
training_set_df.to_csv('training_set.csv', header=False, index=False, encoding='utf-8')
training_set_df.head()


# Repeat the query as above but change outcome of the farm fingerprint hash to collect the remaining 10% of the data for the test set.

# In[26]:


sql="""
WITH site_history as (
  SELECT
      fullVisitorId as visitor_id,
      (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(hits.customDimensions)) AS content_id,
      (SELECT MAX(IF(index=7, value, NULL)) FROM UNNEST(hits.customDimensions)) AS category, 
      (SELECT MAX(IF(index=6, value, NULL)) FROM UNNEST(hits.customDimensions)) AS title,
      (SELECT MAX(IF(index=2, value, NULL)) FROM UNNEST(hits.customDimensions)) AS author_list,
      SPLIT(RPAD((SELECT MAX(IF(index=4, value, NULL)) FROM UNNEST(hits.customDimensions)), 7), '.') as year_month_array,
      LEAD(hits.customDimensions, 1) OVER (PARTITION BY fullVisitorId ORDER BY hits.time ASC) as nextCustomDimensions
  FROM 
    `cloud-training-demos.GA360_test.ga_sessions_sample`,   
     UNNEST(hits) AS hits
   WHERE 
     # only include hits on pages
      hits.type = "PAGE"
      AND
      fullVisitorId IS NOT NULL
      AND
      hits.time != 0
      AND
      hits.time IS NOT NULL
      AND
      (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(hits.customDimensions)) IS NOT NULL
)
SELECT
  visitor_id,
  content_id,
  category,
  REGEXP_REPLACE(title, r",", "") as title,
  REGEXP_EXTRACT(author_list, r"^[^,]+") as author,
  DATE_DIFF(DATE(CAST(year_month_array[OFFSET(0)] AS INT64), CAST(year_month_array[OFFSET(1)] AS INT64), 1), DATE(1970,1,1), MONTH) as months_since_epoch,
  (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(nextCustomDimensions)) as next_content_id
FROM
  site_history
WHERE (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(nextCustomDimensions)) IS NOT NULL
      AND MOD(ABS(FARM_FINGERPRINT(CONCAT(visitor_id, content_id))), 10) >= 9
"""

test_set_df = bq.Query(sql).execute().result().to_dataframe()
test_set_df.to_csv('test_set.csv', header=False, index=False, encoding='utf-8')
test_set_df.head()


# Let's have a look at the two csv files we just created containing the training and test set. We'll also do a line count of both files to confirm that we have achieved an approximate 90/10 train/test split.  
# In the next notebook, **Content Based Filtering** we will build a model to recommend an article given information about the current article being read, such as the category, title, author, and publish date. 

# In[27]:


get_ipython().run_cell_magic('bash', '', 'wc -l *_set.csv')


# In[28]:


get_ipython().system('head *_set.csv')
