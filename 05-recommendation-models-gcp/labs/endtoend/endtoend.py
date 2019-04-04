
# coding: utf-8

# # Recommendations on GCP with TensorFlow and WALS with Cloud Composer
# ***
# This lab is adapted from the original [solution](https://github.com/GoogleCloudPlatform/tensorflow-recommendation-wals) created by [lukmanr](https://github.com/GoogleCloudPlatform/tensorflow-recommendation-wals/commits?author=lukmanr) 

# This project deploys a solution for a recommendation service on GCP, using the WALS algorithm in TensorFlow. Components include:
# 
# - Recommendation model code, and scripts to train and tune the model on ML Engine
# - A REST endpoint using Google Cloud Endpoints for serving recommendations
# - An Airflow server managed by Cloud Composer for running scheduled model training
# 

# ## Confirm Prerequisites

# ### Create a Cloud Composer Instance
# - Create a Cloud Composer [instance](https://console.cloud.google.com/composer/environments/create?project=)
#     1. Specify 'composer' for name
#     2. Choose a location
#     3. Keep the remaining settings at their defaults
#     4. Select Create
# 
# This takes 15 - 20 minutes. Continue with the rest of the lab as you will be using Cloud Composer near the end.

# In[1]:


get_ipython().run_cell_magic('bash', '', 'pip install sh --upgrade pip # needed to execute shell scripts later')


# ### Setup environment variables
# <span style="color: blue">__Replace the below settings with your own.__</span> Note: you can leave AIRFLOW_BUCKET blank and come back to it after your Composer instance is created which automatically will create an Airflow bucket for you. <br><br>
# 
# ### 1. Make a GCS bucket with the name recserve_[YOUR-PROJECT-ID]:

# In[2]:


## in datalabvm cloud datalab notebook:
import os
output = os.popen("gcloud config get-value project").readlines()
project_name = output[0][:-1]

# change these to try this notebook out
PROJECT = project_name
BUCKET = project_name
#BUCKET = BUCKET.replace("qwiklabs-gcp-", "inna-bckt-")
REGION = 'europe-west1'  ## note: Cloud ML Engine not availabe in europe-west3!

print(PROJECT)
print(BUCKET)
print(REGION)


# do not change these
os.environ['PROJECT'] = PROJECT
os.environ['BUCKET'] = 'recserve_' + PROJECT
os.environ['REGION'] = REGION


# In[3]:


get_ipython().run_cell_magic('bash', '', '\ngcloud config set project $PROJECT\ngcloud config set compute/region $REGION')


# In[4]:


get_ipython().run_cell_magic('bash', '', '\n# create GCS bucket with recserve_PROJECT_NAME if not exists\nexists=$(gsutil ls -d | grep -w gs://${BUCKET}/)\nif [ -n "$exists" ]; then\n   echo "Not creating recserve_bucket since it already exists."\nelse\n   echo "Creating recserve_bucket"\n   gsutil mb -l ${REGION} gs://${BUCKET}\nfi')


# ### Setup Google App Engine permissions
# 1. In [IAM](https://console.cloud.google.com/iam-admin/iam?project=), __change permissions for "Compute Engine default service account" from Editor to Owner__. This is required so you can create and deploy App Engine versions from within Cloud Datalab. Note: the alternative is to run all app engine commands directly in Cloud Shell instead of from within Cloud Datalab.<br/><br/>
# 
# 2. Create an App Engine instance if you have not already by uncommenting and running the below code

# In[ ]:


# %%bash
# run app engine creation commands
# gcloud app create --region ${REGION} # see: https://cloud.google.com/compute/docs/regions-zones/
# gcloud app update --no-split-health-checks


# # Part One: Setup and Train the WALS Model

# ## Upload sample data to BigQuery 
# This tutorial comes with a sample Google Analytics data set, containing page tracking events from the Austrian news site Kurier.at. The schema file '''ga_sessions_sample_schema.json''' is located in the folder data in the tutorial code, and the data file '''ga_sessions_sample.json.gz''' is located in a public Cloud Storage bucket associated with this tutorial. To upload this data set to BigQuery:

# ### Copy sample data files into our bucket

# In[5]:


get_ipython().run_cell_magic('bash', '', '\ngsutil -m cp gs://cloud-training-demos/courses/machine_learning/deepdive/10_recommendation/endtoend/data/ga_sessions_sample.json.gz gs://${BUCKET}/data/ga_sessions_sample.json.gz\ngsutil -m cp gs://cloud-training-demos/courses/machine_learning/deepdive/10_recommendation/endtoend/data/recommendation_events.csv data/recommendation_events.csv\ngsutil -m cp gs://cloud-training-demos/courses/machine_learning/deepdive/10_recommendation/endtoend/data/recommendation_events.csv gs://${BUCKET}/data/recommendation_events.csv')


# ### 2. Create empty BigQuery dataset and load sample JSON data
# Note: Ingesting the 400K rows of sample data. This usually takes 5-7 minutes.

# In[6]:


get_ipython().run_cell_magic('bash', '', '\n# create BigQuery dataset if it doesn\'t already exist\nexists=$(bq ls -d | grep -w GA360_test)\nif [ -n "$exists" ]; then\n   echo "Not creating GA360_test since it already exists."\nelse\n   echo "Creating GA360_test dataset."\n   bq --project_id=${PROJECT} mk GA360_test \nfi\n\n# create the schema and load our sample Google Analytics session data\nbq load --source_format=NEWLINE_DELIMITED_JSON \\\n GA360_test.ga_sessions_sample \\\n gs://${BUCKET}/data/ga_sessions_sample.json.gz \\\n data/ga_sessions_sample_schema.json # can\'t load schema files from GCS')


# ## Install WALS model training package and model data

# ### 1. Create a distributable package. Copy the package up to the code folder in the bucket you created previously.

# In[7]:


get_ipython().run_cell_magic('bash', '', '\ncd wals_ml_engine\n\necho "creating distributable package"\npython setup.py sdist\n\necho "copying ML package to bucket"\ngsutil cp dist/wals_ml_engine-0.1.tar.gz gs://${BUCKET}/code/')


# #### `setup.py` file:

# In[8]:


get_ipython().run_cell_magic('bash', '', 'cat wals_ml_engine/setup.py')


# ### 2. Run the WALS model on the sample data set:

# In[9]:


get_ipython().run_cell_magic('bash', '', '\n# view the ML train local script before running\ncat wals_ml_engine/mltrain.sh')


# In[10]:


get_ipython().run_cell_magic('bash', '', '\ncd wals_ml_engine\n\n# train locally with unoptimized hyperparams\n./mltrain.sh local ../data/recommendation_events.csv --data-type web_views --use-optimized\n\n# Options if we wanted to train on CMLE. We will do this with Cloud Composer later\n# train on ML Engine with optimized hyperparams\n# ./mltrain.sh train ../data/recommendation_events.csv --data-type web_views --use-optimized\n\n# tune hyperparams on ML Engine:\n# ./mltrain.sh tune ../data/recommendation_events.csv --data-type web_views')


# This will take a couple minutes, and create a job directory under wals_ml_engine/jobs like "wals_ml_local_20180102_012345/model", containing the model files saved as numpy arrays.

# ### View the locally trained model directory

# In[11]:


ls wals_ml_engine/jobs


# ### 3. Copy the model files from this directory to the model folder in the project bucket:
# In the case of multiple models, take the most recent (tail -1)

# In[12]:


get_ipython().run_cell_magic('bash', '', 'export JOB_MODEL=$(find wals_ml_engine/jobs -name "model" | tail -1)\ngsutil cp ${JOB_MODEL}/* gs://${BUCKET}/model/\n  \necho "Recommendation model file numpy arrays in bucket:"  \ngsutil ls gs://${BUCKET}/model/')


# # Install the recserve endpoint

# ### 1. Prepare the deploy template for the Cloud Endpoint API:

# In[17]:


get_ipython().run_line_magic('writefile', 'scripts/prepare_deploy_api.sh')
#!/bin/bash
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

source util.sh

main() {
  # Get our working project, or exit if it's not set.
  local project_id=$(get_project_id)
  if [[ -z "$project_id" ]]; then
    exit 1
  fi
  local temp_file=$(mktemp)
  export TEMP_FILE="${temp_file}.yaml"
  mv "$temp_file" "$TEMP_FILE"

  # Because the included API is a template, we have to do some string
  # substitution before we can deploy it. Sed does this nicely.
  < "$API_FILE" sed -E "s/YOUR-PROJECT-ID/${project_id}/g" > "$TEMP_FILE"
  echo "Preparing config for deploying service in $API_FILE..."
  echo "To deploy:  gcloud endpoints services deploy $TEMP_FILE"
}

# Defaults.
API_FILE="../app/openapi.yaml"

if [[ "$#" == 0 ]]; then
  : # Use defaults.
elif [[ "$#" == 1 ]]; then
  API_FILE="$1"
else
  echo "Wrong number of arguments specified."
  echo "Usage: deploy_api.sh [api-file]"
  exit 1
fi

main "$@"


# In[23]:


get_ipython().run_cell_magic('bash', '', 'ls\ncd scripts\ncat prepare_deploy_api.sh')


# In[26]:


get_ipython().run_line_magic('writefile', 'scripts/util.sh')
#!/bin/bash
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Make Bash a little less error-prone.
set -euo pipefail

get_latest_config_id() {
  # Given a service name, this returns the most recent deployment of that
  # API.
  service_name="$1"
  gcloud endpoints configs list \
    --service="$service_name" \
    --sort-by="~config_id" --limit=1 --format="value(CONFIG_ID)" \
    | tr -d '[:space:]'
}

get_project_id() {
  # Find the project ID first by DEVSHELL_PROJECT_ID (in Cloud Shell)
  # and then by querying the gcloud default project.
  local project="${DEVSHELL_PROJECT_ID:-}"
  if [[ -z "$project" ]]; then
    project=$(gcloud config get-value project 2> /dev/null)
  fi
  if [[ -z "$project" ]]; then
    >&2 echo "No default project was found, and DEVSHELL_PROJECT_ID is not set."
    >&2 echo "Please use the Cloud Shell or set your default project by typing:"
    >&2 echo "gcloud config set project YOUR-PROJECT-NAME"
  fi
  echo "$project"
}


# In[27]:


get_ipython().run_cell_magic('bash', '', 'printf "\\nCopy and run the deploy script generated below:\\n"\ncd scripts\nls\nchmod +777 ./prepare_deploy_api.sh\n./prepare_deploy_api.sh                         # Prepare config file for the API.')


# This will output somthing like:
# 
# ```To deploy:  gcloud endpoints services deploy /var/folders/1m/r3slmhp92074pzdhhfjvnw0m00dhhl/T/tmp.n6QVl5hO.yaml```

# ### 2. Run the endpoints deploy command output above:
# <span style="color: blue">Be sure to __replace the below [FILE_NAME]__ with the results from above before running.</span>

# In[29]:


get_ipython().run_cell_magic('bash', '', 'gcloud endpoints services deploy /tmp/tmp.GRLr0VWjcf.yaml')


# ### 3. Prepare the deploy template for the App Engine App:

# In[30]:


get_ipython().run_line_magic('writefile', 'scripts/prepare_deploy_app.sh')
#!/bin/bash
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

source util.sh

main() {
  # Get our working project, or exit if it's not set.
  local project_id="$(get_project_id)"
  if [[ -z "$project_id" ]]; then
    exit 1
  fi
  # Try to create an App Engine project in our selected region.
  # If it already exists, return a success ("|| true").
  echo "gcloud app create --region=$REGION"
  gcloud app create --region="$REGION" || true

  # Prepare the necessary variables for substitution in our app configuration
  # template, and create a temporary file to hold the templatized version.
  local service_name="${project_id}.appspot.com"
  local config_id=$(get_latest_config_id "$service_name")
  export TEMP_FILE="${APP}_deploy.yaml"
  < "$APP"     sed -E "s/SERVICE_NAME/${service_name}/g"     | sed -E "s/SERVICE_CONFIG_ID/${config_id}/g"     > "$TEMP_FILE"

  echo "To deploy:  gcloud -q app deploy $TEMP_FILE"
}

# Defaults.
APP="../app/app_template.yaml"
REGION="us-east1"
SERVICE_NAME="default"

if [[ "$#" == 0 ]]; then
  : # Use defaults.
elif [[ "$#" == 1 ]]; then
  APP="$1"
elif [[ "$#" == 2 ]]; then
  APP="$1"
  REGION="$2"
else
  echo "Wrong number of arguments specified."
  echo "Usage: deploy_app.sh [app-template] [region]"
  exit 1
fi

main "$@"


# In[31]:


get_ipython().run_cell_magic('bash', '', '# prepare to deploy \ncd scripts\nchmod +777 ./prepare_deploy_app.sh\n./prepare_deploy_app.sh')


# You can ignore the script output "ERROR: (gcloud.app.create) The project [...] already contains an App Engine application. You can deploy your application using gcloud app deploy." This is expected.
# 
# The script will output something like:
# 
# ```To deploy:  gcloud -q app deploy app/app_template.yaml_deploy.yaml```

# ### 4. Run the command above:

# In[33]:


get_ipython().run_cell_magic('bash', '', 'gcloud -q app deploy app/app_template.yaml_deploy.yaml')


# This will take 7 - 10 minutes to deploy the app. While you wait, consider starting on Part Two below and completing the Cloud Composer DAG file.

# ## Query the API for Article Recommendations
# Lastly, you are able to test the recommendation model API by submitting a query request. Note the example userId passed and numRecs desired as the URL parameters for the model input.

# In[34]:


get_ipython().run_line_magic('writefile', 'scripts/query_api.sh')
#!/bin/bash
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

source util.sh

main() {
  # Get our working project, or exit if it's not set.
  local project_id=$(get_project_id)
  if [[ -z "$project_id" ]]; then
    exit 1
  fi
  # Because our included app uses query string parameters, we can include
  # them directly in the URL.
  QUERY="curl \"https://${project_id}.appspot.com/recommendation?userId=${USER_ID}&numRecs=${NUM_RECS}\""
  # First, (maybe) print the command so the user can see what's being executed.
  if [[ "$QUIET" == "false" ]]; then
    echo "$QUERY"
  fi
  # Then actually execute it.
  # shellcheck disable=SC2086
  eval $QUERY
  # Our API doesn't print newlines. So we do it ourselves.
  printf '\n'
}

# Defaults.
USER_ID="5448543647176335931"
NUM_RECS=5
QUIET="false"

if [[ "$#" == 0 ]]; then
  : # Use defaults.
elif [[ "$#" == 1 ]]; then
  USER_ID="$1"
elif [[ "$#" == 2 ]]; then
  USER_ID="$1"
  NUM_RECS="$2"
elif [[ "$#" == 3 ]]; then
  # "Quiet mode" won't print the curl command.
  USER_ID="$1"
  NUM_RECS="$2"
  QUIET="true"
else
  echo "Wrong number of arguments specified."
  echo "Usage: query_api.sh [user-id] [num-recs] [quiet-mode]"
  exit 1
fi

main "$@"


# In[35]:


get_ipython().run_line_magic('writefile', 'scripts/generate_traffic.sh')
#!/bin/bash
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

source util.sh

main() {
  # Get our working project, or exit if it's not set.
  local project_id=$(get_project_id)
  if [[ -z "$project_id" ]]; then
    exit 1
  fi
  local url="https://${project_id}.appspot.com/recommendation?userId=${USER_ID}&numRecs=${NUM_RECS}"
  echo "This command will exit automatically in $TIMEOUT_SECONDS seconds."
  echo "Generating traffic to ${url}..."
  echo "Press Ctrl-C to stop."
  local endtime=$(($(date +%s) + $TIMEOUT_SECONDS))
  # Send queries repeatedly until TIMEOUT_SECONDS seconds have elapsed.
  while [[ $(date +%s) -lt $endtime ]]; do
    curl "$url" &> /dev/null
  done
}

# Defaults.
USER_ID="5448543647176335931"
NUM_RECS=5
TIMEOUT_SECONDS=$((5 * 60)) # Timeout after 5 minutes.

if [[ "$#" == 0 ]]; then
  : # Use defaults.
elif [[ "$#" == 1 ]]; then
  USER_ID="$1"
else
  echo "Wrong number of arguments specified."
  echo "Usage: generate_traffic.sh [user_id]"
  exit 1
fi

main "$@"


# In[36]:


get_ipython().run_cell_magic('bash', '', 'cd scripts\nchmod +777 ./query_api.sh     \nchmod +777 ./generate_traffic.sh\n./query_api.sh          # Query the API.\n./generate_traffic.sh   # Send traffic to the API.')


# If the call is successful, you will see the article IDs recommended for that specific user by the WALS ML model <br/>
# (Example: curl "https://qwiklabs-gcp-12345.appspot.com/recommendation?userId=5448543647176335931&numRecs=5"
# {"articles":["299824032","1701682","299935287","299959410","298157062"]} )
# 
# __Part One is done!__ You have successfully created the back-end architecture for serving your ML recommendation system. But we're not done yet, we still need to automatically retrain and redeploy our model once new data comes in. For that we will use [Cloud Composer](https://cloud.google.com/composer/) and [Apache Airflow](https://airflow.apache.org/).<br/><br/>

# ***
# # Part Two: Setup a scheduled workflow with Cloud Composer
# In this section you will complete a partially written training.py DAG file and copy it to the DAGS folder in your Composer instance.

# ## Copy your Airflow bucket name
# 1. Navigate to your Cloud Composer [instance](https://console.cloud.google.com/composer/environments?project=)<br/><br/>
# 2. Select __DAGs Folder__<br/><br/>
# 3. You will be taken to the Google Cloud Storage bucket that Cloud Composer has created automatically for your Airflow instance<br/><br/>
# 4. __Copy the bucket name__ into the variable below (example: us-central1-composer-08f6edeb-bucket)

# In[37]:


AIRFLOW_BUCKET = 'europe-west1-mlcomposer-aaa81fd3-bucket' # REPLACE WITH AIRFLOW BUCKET NAME
os.environ['AIRFLOW_BUCKET'] = AIRFLOW_BUCKET


# ## Complete the training.py DAG file
# Apache Airflow orchestrates tasks out to other services through a [DAG (Directed Acyclic Graph)](https://airflow.apache.org/concepts.html) file which specifies what services to call, what to do, and when to run these tasks. DAG files are written in python and are loaded automatically into Airflow once present in the Airflow/dags/ folder in your Cloud Composer bucket. 
# 
# Your task is to complete the partially written DAG file below which will enable the automatic retraining and redeployment of our WALS recommendation model. 
# 
# __Complete the #TODOs__ in the Airflow DAG file below and execute the code block to save the file

# In[38]:


get_ipython().run_cell_magic('writefile', 'airflow/dags/training.py', '\n# Copyright 2018 Google Inc. All Rights Reserved.\n#\n# Licensed under the Apache License, Version 2.0 (the "License");\n# you may not use this file except in compliance with the License.\n# You may obtain a copy of the License at\n#\n# http://www.apache.org/licenses/LICENSE-2.0\n#\n# Unless required by applicable law or agreed to in writing, software\n# distributed under the License is distributed on an "AS IS" BASIS,\n# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n# See the License for the specific language governing permissions and\n# limitations under the License.\n\n"""DAG definition for recserv model training."""\n\nimport airflow\nfrom airflow import DAG\n\n# Reference for all available airflow operators: \n# https://github.com/apache/incubator-airflow/tree/master/airflow/contrib/operators\nfrom airflow.contrib.operators.bigquery_operator import BigQueryOperator\nfrom airflow.contrib.operators.bigquery_to_gcs import BigQueryToCloudStorageOperator\nfrom airflow.hooks.base_hook import BaseHook\n# from airflow.contrib.operators.mlengine_operator import MLEngineTrainingOperator\n# above mlengine_operator currently doesnt support custom MasterType so we import our own plugins:\n\n# custom plugins\nfrom airflow.operators.app_engine_admin_plugin import AppEngineVersionOperator\nfrom airflow.operators.ml_engine_plugin import MLEngineTrainingOperator\n\n\nimport datetime\n\ndef _get_project_id():\n  """Get project ID from default GCP connection."""\n\n  extras = BaseHook.get_connection(\'google_cloud_default\').extra_dejson\n  key = \'extra__google_cloud_platform__project\'\n  if key in extras:\n    project_id = extras[key]\n  else:\n    raise (\'Must configure project_id in google_cloud_default \'\n           \'connection from Airflow Console\')\n  return project_id\n\nPROJECT_ID = _get_project_id()\n\n# Data set constants, used in BigQuery tasks.  You can change these\n# to conform to your data.\n\n# TODO (done): Specify your BigQuery dataset name and table name\nDATASET = \'GA360_test\'\nTABLE_NAME = \'ga_sessions_sample\'\nARTICLE_CUSTOM_DIMENSION = \'10\'\n\n# TODO (done): Confirm bucket name and region\n# GCS bucket names and region, can also be changed.\nBUCKET = \'gs://recserve_\' + PROJECT_ID\nREGION = \'europe-west1\'\n\n# The code package name comes from the model code in the wals_ml_engine\n# directory of the solution code base.\nPACKAGE_URI = BUCKET + \'/code/wals_ml_engine-0.1.tar.gz\'\nJOB_DIR = BUCKET + \'/jobs\'\n\ndefault_args = {\n    \'owner\': \'airflow\',\n    \'depends_on_past\': False,\n    \'start_date\': airflow.utils.dates.days_ago(2),\n    \'email\': [\'airflow@example.com\'],\n    \'email_on_failure\': True,\n    \'email_on_retry\': False,\n    \'retries\': 5,\n    \'retry_delay\': datetime.timedelta(minutes=5)\n}\n\n# Default schedule interval using cronjob syntax - can be customized here\n# or in the Airflow console.\n\n# TODO (done): Specify a schedule interval in CRON syntax to run once a day at 2100 hours (9pm)\n# Reference: https://airflow.apache.org/scheduler.html\nschedule_interval = \'00 21 * * *\' # example \'00 XX 0 0 0\'\n\n# TODO (done): Title your DAG to be recommendations_training_v1\ndag = DAG(\'recommendations_training_v1\', \n          default_args=default_args,\n          schedule_interval=schedule_interval)\n\ndag.doc_md = __doc__\n\n\n#\n#\n# Task Definition\n#\n#\n\n# BigQuery training data query\n\nbql=\'\'\'\n#legacySql\nSELECT\n fullVisitorId as clientId,\n ArticleID as contentId,\n (nextTime - hits.time) as timeOnPage,\nFROM(\n  SELECT\n    fullVisitorId,\n    hits.time,\n    MAX(IF(hits.customDimensions.index={0},\n           hits.customDimensions.value,NULL)) WITHIN hits AS ArticleID,\n    LEAD(hits.time, 1) OVER (PARTITION BY fullVisitorId, visitNumber\n                             ORDER BY hits.time ASC) as nextTime\n  FROM [{1}.{2}.{3}]\n  WHERE hits.type = "PAGE"\n) HAVING timeOnPage is not null and contentId is not null;\n\'\'\'\n\nbql = bql.format(ARTICLE_CUSTOM_DIMENSION, PROJECT_ID, DATASET, TABLE_NAME)\n\n# TODO (done): Complete the BigQueryOperator task to truncate the table if it already exists before writing\n# Reference: https://airflow.apache.org/integration.html#bigqueryoperator\nt1 = BigQueryOperator( # correct the operator name\n    task_id=\'bq_rec_training_data\',\n    bql=bql,\n    destination_dataset_table=\'%s.recommendation_events\' % DATASET,\n    write_disposition=\'WRITE_TRUNCATE\', # specify to truncate on writes\n    dag=dag)\n\n# BigQuery training data export to GCS\n\n# TODO (done): Fill in the missing operator name for task #2 which\n# takes a BigQuery dataset and table as input and exports it to GCS as a CSV\ntraining_file = BUCKET + \'/data/recommendation_events.csv\'\nt2 = BigQueryToCloudStorageOperator( # correct the name\n    task_id=\'bq_export_op\',\n    source_project_dataset_table=\'%s.recommendation_events\' % DATASET,\n    destination_cloud_storage_uris=[training_file],\n    export_format=\'CSV\',\n    dag=dag\n)\n\n\n# ML Engine training job\n\njob_id = \'recserve_{0}\'.format(datetime.datetime.now().strftime(\'%Y%m%d%H%M\'))\njob_dir = BUCKET + \'/jobs/\' + job_id\noutput_dir = BUCKET\ntraining_args = [\'--job-dir\', job_dir,\n                 \'--train-files\', training_file,\n                 \'--output-dir\', output_dir,\n                 \'--data-type\', \'web_views\',\n                 \'--use-optimized\']\n\n# TODO (done): Fill in the missing operator name for task #3 which will\n# start a new training job to Cloud ML Engine\n# Reference: https://airflow.apache.org/integration.html#cloud-ml-engine\n# https://cloud.google.com/ml-engine/docs/tensorflow/machine-types\nt3 = MLEngineTrainingOperator( # complete the name\n    task_id=\'ml_engine_training_op\',\n    project_id=PROJECT_ID,\n    job_id=job_id,\n    package_uris=[PACKAGE_URI],\n    training_python_module=\'trainer.task\',\n    training_args=training_args,\n    region=REGION,\n    scale_tier=\'CUSTOM\',\n    master_type=\'complex_model_m_gpu\',\n    dag=dag\n)\n\n# App Engine deploy new version\n\nt4 = AppEngineVersionOperator(\n    task_id=\'app_engine_deploy_version\',\n    project_id=PROJECT_ID,\n    service_id=\'default\',\n    region=REGION,\n    service_spec=None,\n    dag=dag\n)\n\n# TODO (done): Be sure to set_upstream dependencies for all tasks\nt2.set_upstream(t1)\nt3.set_upstream(t2)\nt4.set_upstream(t3) # complete')


# ### Copy local Airflow DAG file and plugins into the DAGs folder

# In[39]:


get_ipython().run_cell_magic('bash', '', 'gsutil cp airflow/dags/training.py gs://${AIRFLOW_BUCKET}/dags # overwrite if it exists\ngsutil cp -r airflow/plugins gs://${AIRFLOW_BUCKET} # copy custom plugins')


# 2. Navigate to your Cloud Composer [instance](https://console.cloud.google.com/composer/environments?project=)<br/><br/>
# 
# 3. Trigger a __manual run__ of your DAG for testing<br/><br/>
# 
# 3. Ensure your DAG runs successfully (all nodes outlined in dark green and 'success' tag shows)
# 
# ![Successful Airflow DAG run](./img/airflow_successful_run.jpg "Successful Airflow DAG run")
# 

# ## Troubleshooting your DAG
# 
# DAG not executing successfully? Follow these below steps to troubleshoot.
# 
# Click on the name of a DAG to view a run (ex: recommendations_training_v1)
# 
# 1. Select a node in the DAG (red or yellow borders mean failed nodes)
# 2. Select View Log
# 3. Scroll to the bottom of the log to diagnose
# 4. X Option: Clear and immediately restart the DAG after diagnosing the issue
# 
# Tips:
# - If bq_rec_training_data immediately fails without logs, your DAG file is missing key parts and is not compiling
# - ml_engine_training_op will take 9 - 12 minutes to run. Monitor the training job in [ML Engine](https://console.cloud.google.com/mlengine/jobs?project=)
# - Lastly, check the [solution endtoend.ipynb](../endtoend/endtoend.ipynb) to compare your lab answers

# ![Viewing Airflow logs](./img/airflow_viewing_logs.jpg "Viewing Airflow logs")

# # Congratulations!
# You have made it to the end of the end-to-end recommendation system lab. You have successfully setup an automated workflow to retrain and redeploy your recommendation model.

# ***
# # Challenges
# 
# Looking to solidify your Cloud Composer skills even more? Complete the __optional challenges__ below
# <br/><br/>
# ### Challenge 1
# Use either the [BigQueryCheckOperator](https://airflow.apache.org/integration.html#bigquerycheckoperator) or the [BigQueryValueCheckOperator](https://airflow.apache.org/integration.html#bigqueryvaluecheckoperator) to create a new task in your DAG that ensures the SQL query for training data is returning valid results before it is passed to Cloud ML Engine for training. 
# <br/><br/>
# Hint: Check for COUNT() = 0 or other health check
# <br/><br/><br/>
# ### Challenge 2
# Create a Cloud Function to [automatically trigger](https://cloud.google.com/composer/docs/how-to/using/triggering-with-gcf) your DAG when a new recommendation_events.csv file is loaded into your Google Cloud Storage Bucket. 
# <br/><br/>
# Hint: Check the [composer_gcf_trigger.ipynb lab](../composer_gcf_trigger/composertriggered.ipynb) for inspiration
# <br/><br/><br/>
# ### Challenge 3
# Modify the BigQuery query in the DAG to only train on a portion of the data available in the dataset using a WHERE clause filtering on date. Next, parameterize the WHERE clause to be based on when the Airflow DAG is run
# <br/><br/>
# Hint: Make use of prebuilt [Airflow macros](https://airflow.incubator.apache.org/_modules/airflow/macros.html) like the below:
# 
# _constants or can be dynamic based on Airflow macros_ <br/>
# max_query_date = '2018-02-01' # {{ macros.ds_add(ds, -7) }} <br/>
# min_query_date = '2018-01-01' # {{ macros.ds_add(ds, -1) }} 
# 

# ## Additional Resources
# 
# - Follow the latest [Airflow operators](https://github.com/apache/incubator-airflow/tree/master/airflow/contrib/operators) on github