# Code Snippets for End-to-End Machine Learning with TensorFlow on GCP

## [[todo]]

* add solutions to my github repo
  * maybe also as pdf's?
* add email address config to git in docker container (for contributions)
* install jupyterlab in docker container and run it on port that can be previewed (8082?)

## [[?]]




## Cloud Shell

### get project id

```bash
gcloud config list
gcloud config get-value project
gcloud config get-value account
```



### set environment variables

```bash
export PROJECT=$(gcloud config get-value project)
echo $PROJECT
```

### Start datalab VM

```bash
gcloud compute zones list
datalab create mydatalabvm --zone europe-west1-c
```

If it times out, just reconnect:

```bash
datalab connect mydatalabvm
```



### SSH into cloud VM instance

```bash
gcloud compute --project <project-id> ssh --zone <zone> <instance-name>

gcloud compute --project qwiklabs-gcp-0e6fd357ee5ebeda ssh --zone europe-west1-c mydatalabvm
```
### Execute interactive shell session in datalab docker container

* First, ssh into the cloud VM instance for the datalab
* then:
```bash
sudo docker ps
#docker exec -it <container-id> bash
docker exec -it 7b405cbff6db bash
cd /content/datalab/
```
### Git in Cloud VM instance

* SSH into cloudvm instance
  * either by selecting the SSH button in the VM instances list (GCP sandwich menu)
  * or by running command line above
* then execute an interactive bash session in the docker container that runs the lab:

```bash
cd /content/datalab/
git clone https://github.com/GoogleCloudPlatform/training-data-analyst
rm -rf training-data-analyst/.git
```

```bash
cd /content/datalab/
git clone https://github.com/ingonader/gcp-ml-02-advanced-ml-with-tf-on-gcp.git
git config user.email "ingo.nader@gmail.com"
#git config --global user.email "ingo.nader@gmail.com"
```

## Python / Data Lab

### get project id in python / data lab

```python
import os
output = os.popen("gcloud config get-value project").readlines()
project_name = output[0]
print(project_name)

# change these to try this notebook out
BUCKET = project_name
PROJECT = project_name
REGION = 'eu-west3'
```

## get number of rows for query

```python
import google.datalab.bigquery as bq
query = """
SELECT
  weight_pounds,
  is_male,
  mother_age,
  plurality,
  gestation_weeks,
  ABS(FARM_FINGERPRINT(CONCAT(CAST(YEAR AS STRING), CAST(month AS STRING)))) AS hashmonth
FROM
  publicdata.samples.natality
WHERE year > 2000
"""
## count rows
query_rowcount = bq.Query("select count(*) from (" + query + ")").execute().result().to_dataframe()
print(query_rowcount) 

## sample from query (limit data traffic, but not costs):
dat_query_sample = bq.Query(query + " LIMIT 100").execute().result().to_dataframe()
dat_query_sample.head(n = 10)
```



## use alias in `where` clause

* alias can only be used via a subquery!

```python
query = """
SELECT * FROM 
(
SELECT
  weight_pounds,
  is_male,
  mother_age,
  plurality,
  gestation_weeks,
  ABS(FARM_FINGERPRINT(CONCAT(CAST(YEAR AS STRING), CAST(month AS STRING)))) AS hashmonth
FROM
  publicdata.samples.natality
WHERE year > 2000
) 
WHERE MOD(hashmonth, 10) < 8  -- alias can only be used via subquery
```

