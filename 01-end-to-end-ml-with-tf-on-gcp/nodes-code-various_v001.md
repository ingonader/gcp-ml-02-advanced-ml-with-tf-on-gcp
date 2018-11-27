# Code Snippets for End-to-End Machine Learning with TensorFlow on GCP

## [[todo]]

* install jupyterlab in docker container and run it on port that can be previewed (8082?)

```bash
jupyter notebook --version ## requires >= 4.3
pip install jupyterlab
jupyter lab
```

* Find out docker mappings (no new ports can be added to running container):

```bash
docker port <CONTAINER>

## or:
docker inspect --format='{{range $p, $conf := .NetworkSettings.Ports}} {{$p}} -> {{(index $conf 0).HostPort}} {{end}}' $INSTANCE_ID
```

* Find out why `gsutil mb -l ${REGION} gs://${BUCKET}` in python script produces `BadRequestException: 400 Invalid Value` error.
* Make `pdf`s and `html`s of already run notebooks!



## [[?]]

* "feature cross all the wide columns [[?]]"

  ```python
  # Feature cross all the wide columns and embed into a lower dimension
  crossed = tf.feature_column.crossed_column(wide, hash_bucket_size=20000)
  embed = tf.feature_column.embedding_column(crossed, 3)
  ```

  * What does this mean and do?

* Apache Beam:

  * What is a `DirectRunner`?  Anwer: runs locally
  * What is the difference to a `DataflowRunner`? Answer: used to run dataflow job in the cloud.

* 

## Most commonly used code snippets (copied from below)

http://console.cloud.google.com/

```bash
## locally:
jupyter notebook list
gsutil cp gs://my-bucket/my-file .

## in cloud shell:

## create datalab vm:
export PROJECT=$(gcloud config get-value project)
export ZONE=europe-west1-c
# echo "Y" | datalab create mydatalabvm --zone $ZONE
printf "Y\n\n\n" | datalab create mydatalabvm --zone $ZONE
# datalab create mydatalabvm --zone $ZONE

## ssh into datalab vm:
#export PROJECT=$(gcloud config get-value project)
#export ZONE=europe-west1-c
gcloud compute ssh mydatalabvm --project $PROJECT --zone $ZONE

## in datalab vm:

## run interactive bash in docker container:
sudo docker ps
export CONTAINER=datalab
docker exec -it $CONTAINER bash

## clone git repo:
cd /content/datalab/
git clone https://github.com/ingonader/gcp-ml-02-advanced-ml-with-tf-on-gcp.git
cd gcp-ml-02-advanced-ml-with-tf-on-gcp
git config user.email "ingo.nader@gmail.com"
```

```python
## in datalabvm cloud datalab notebook:
import os
output = os.popen("gcloud config get-value project").readlines()
project_name = output[0][:-1]

# change these to try this notebook out
PROJECT = project_name
BUCKET = project_name
BUCKET = BUCKET.replace("qwiklabs-gcp-", "inna-bckt-")
REGION = 'europe-west1'  ## note: Cloud ML Engine not availabe in europe-west3!

print(PROJECT)
print(BUCKET)
print("gsutil mb -l {0} gs://{1}".format(REGION, BUCKET))

## set config for gcp config: [[?]]
print(os.popen("gcloud config set project $PROJECT").readlines())
print(os.popen("gcloud config set compute/region $REGION").readlines())
```

```bash
## set project in google datalab:
## (might be necessary to use gsutil command?)
%bash
gcloud config set project $PROJECT
gcloud config set compute/region $REGION

## inspect data in cloud storage bucket:
%bash
gsutil cat -r -512 gs://${BUCKET}/babyweight/preproc/eval.csv-00000-of-*
```



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
gcp-ml-02-advanced-ml-with-tf-on-gcp
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



## various

### Change indentation in Juypter Notebooks

  ```javascript
var cell = Jupyter.notebook.get_selected_cell();
var config = cell.config;
var patch = {
      CodeCell:{
        cm_config:{indentUnit:2}
      }
    }
config.update(patch)
  ```
  Enter the previous snippet in your browser’s 
  **JavaScript console** once. Then **reload** the notebook page 
  in your browser. Now, the preferred indent unit should 
  be equal to two spaces. The custom setting persists and 
  you do not need to reissue the patch on new notebooks.