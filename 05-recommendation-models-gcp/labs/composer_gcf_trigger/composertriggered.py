
# coding: utf-8

# # Triggering a Cloud Composer Pipeline with a Google Cloud Function
# 
# In this advanced lab you will learn how to create and run an [Apache Airflow](http://airflow.apache.org/) workflow in Cloud Composer that completes the following tasks:
# - Watches for new CSV data to be uploaded to a [Cloud Storage](https://cloud.google.com/storage/docs/) bucket
# - A [Cloud Function](https://cloud.google.com/composer/docs/how-to/using/triggering-with-gcf#getting_the_client_id) call triggers the [Cloud Composer Airflow DAG](https://cloud.google.com/composer/docs/how-to/using/writing-dags) to run when a new file is detected 
# - The workflow finds the input file that triggered the workflow and executes a [Cloud Dataflow](https://cloud.google.com/dataflow/) job to transform and output the data to BigQuery  
# - Moves the original input file to a different Cloud Storage bucket for storing processed files

# ## Part One: Create Cloud Composer environment and workflow
# First, create a Cloud Composer environment if you don't have one already by doing the following:
# 1. In the **Navigation menu** under Big Data, select **Composer**
# 2. Select **Create**
# 3. Set the following parameters:
#     - Name: mlcomposer
#     - Location: us-central1
#     - Other values at defaults
# 4. Select **Create**
# 
# The environment creation process is completed when the green checkmark displays to the left of the environment name on the Environments page in the GCP Console.
# It can take up to 20 minutes for the environment to complete the setup process. Move on to the next section - Create Cloud Storage buckets and BigQuery dataset.
# 

# ## Set environment variables

# In[1]:


import os
output = os.popen("gcloud config get-value project").readlines()
project_name = output[0][:-1]

PROJECT = project_name
BUCKET = project_name
#BUCKET = BUCKET.replace("qwiklabs-gcp-", "inna-bckt-")
REGION = 'europe-west1'  ## note: Cloud ML Engine not availabe in europe-west3!

print(PROJECT)
print(BUCKET)
print(REGION)

# set environment variables:
os.environ['BUCKET'] = BUCKET
os.environ['PROJECT'] = PROJECT
os.environ['REGION'] = REGION


# ## Create Cloud Storage buckets
# Create two Cloud Storage Multi-Regional buckets in your project. 
# - project-id_input
# - project-id_output
# 
# Run the below to automatically create the buckets:

# In[2]:


get_ipython().run_cell_magic('bash', '', '## create GCS buckets\nexists=$(gsutil ls -d | grep -w gs://${PROJECT}_input/)\nif [ -n "$exists" ]; then\n   echo "Skipping the creation of input bucket."\nelse\n   echo "Creating input bucket."\n   gsutil mb -l ${REGION} gs://${PROJECT}_input\nfi\n\nexists=$(gsutil ls -d | grep -w gs://${PROJECT}_output/)\nif [ -n "$exists" ]; then\n   echo "Skipping the creation of output bucket."\nelse\n   echo "Creating output bucket."\n   gsutil mb -l ${REGION} gs://${PROJECT}_output\nfi')


# ## Create BigQuery Destination Dataset and Table
# Next, we'll create a data sink to store the ingested data from GCS<br><br>
# 
# ### Create a new Dataset
# 1. In the **Navigation menu**, select **BigQuery**
# 2. Then click on your qwiklabs project ID
# 3. Click **Create Dataset**
# 4. Name your dataset **ml_pipeline** and leave other values at defaults
# 5. Click **Create Dataset**
# 
# 
# ### Create a new empty table
# 1. Click on the newly created dataset
# 2. Click **Create Table**
# 3. For Destination Table name specify **ingest_table**
# 4. For schema click **Edit as Text** and paste in the below schema
# 
#     state:  STRING,<br>
#     gender: STRING,<br>
#     year: STRING,<br>
#     name: STRING,<br>
#     number: STRING,<br>
#     created_date: STRING,<br>
#     filename: STRING,<br>
#     load_dt:  DATE<br><br>
# 
# 5. Click **Create Table**

# ## Review of Airflow concepts
# While your Cloud Composer environment is building, let’s discuss the sample file you’ll be using in this lab.
# <br><br>
# [Airflow](https://airflow.apache.org/) is a platform to programmatically author, schedule and monitor workflows
# <br><br>
# Use airflow to author workflows as directed acyclic graphs (DAGs) of tasks. The airflow scheduler executes your tasks on an array of workers while following the specified dependencies.
# <br><br>
# ### Core concepts
# - [DAG](https://airflow.apache.org/concepts.html#dags) - A Directed Acyclic Graph  is a collection of tasks, organised to reflect their relationships and dependencies.
# - [Operator](https://airflow.apache.org/concepts.html#operators) - The description of a single task, it is usually atomic. For example, the BashOperator is used to execute bash command.
# - [Task](https://airflow.apache.org/concepts.html#tasks) - A parameterised instance of an Operator;  a node in the DAG.
# - [Task Instance](https://airflow.apache.org/concepts.html#task-instances) - A specific run of a task; characterised as: a DAG, a Task, and a point in time. It has an indicative state: *running, success, failed, skipped, …*<br><br>
# The rest of the Airflow concepts can be found [here](https://airflow.apache.org/concepts.html#).
# 
# 

# ## Complete the DAG file
# Cloud Composer workflows are comprised of [DAGs (Directed Acyclic Graphs)](https://airflow.incubator.apache.org/concepts.html#dags). The code shown in simple_load_dag.py is the workflow code, also referred to as the DAG. 
# <br><br>
# Open the file now to see how it is built. Next will be a detailed look at some of the key components of the file.
# <br><br>
# To orchestrate all the workflow tasks, the DAG imports the following operators:
# - DataFlowPythonOperator
# - PythonOperator
# <br><br>
# Action: <span style="color:blue">**Complete the # TODOs in the simple_load_dag.py DAG file below**</span> file while you wait for your Composer environment to be setup. 

# In[3]:


get_ipython().run_cell_magic('writefile', 'simple_load_dag.py', '# Copyright 2018 Google LLC\n#\n# Licensed under the Apache License, Version 2.0 (the "License");\n# you may not use this file except in compliance with the License.\n# You may obtain a copy of the License at\n#\n#     https://www.apache.org/licenses/LICENSE-2.0\n#\n# Unless required by applicable law or agreed to in writing, software\n# distributed under the License is distributed on an "AS IS" BASIS,\n# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n# See the License for the specific language governing permissions and\n# limitations under the License.\n\n"""A simple Airflow DAG that is triggered externally by a Cloud Function when a\nfile lands in a GCS bucket.\nOnce triggered the DAG performs the following steps:\n1. Triggers a Google Cloud Dataflow job with the input file information received\n   from the Cloud Function trigger.\n2. Upon completion of the Dataflow job, the input file is moved to a\n   gs://<target-bucket>/<success|failure>/YYYY-MM-DD/ location based on the\n   status of the previous step.\n"""\n\nimport datetime\nimport logging\nimport os\n\nfrom airflow import configuration\nfrom airflow import models\nfrom airflow.contrib.hooks import gcs_hook\nfrom airflow.contrib.operators import dataflow_operator\nfrom airflow.operators import python_operator\nfrom airflow.utils.trigger_rule import TriggerRule\n\n# We set the start_date of the DAG to the previous date. This will\n# make the DAG immediately available for scheduling.\nYESTERDAY = datetime.datetime.combine(\n    datetime.datetime.today() - datetime.timedelta(1),\n    datetime.datetime.min.time())\n\n# We define some variables that we will use in the DAG tasks.\nSUCCESS_TAG = \'success\'\nFAILURE_TAG = \'failure\'\n\n# An Airflow variable called gcp_completion_bucket is required.\n# This variable will contain the name of the bucket to move the processed\n# file to.\n\n# \'_names\' must appear in CSV filename to be ingested (adjust as needed)\n# we are only looking for files with the exact name usa_names.csv (you can specify wildcards if you like)\nINPUT_BUCKET_CSV = \'gs://\'+models.Variable.get(\'gcp_input_location\')+\'/usa_names.csv\' \n\n# TODO (done): Populate the models.Variable.get() with the actual variable name for your output bucket\nCOMPLETION_BUCKET = \'gs://\'+models.Variable.get(\'gcp_output_location\')\n\nDS_TAG = \'{{ ds }}\'\nDATAFLOW_FILE = os.path.join(\n    configuration.get(\'core\', \'dags_folder\'), \'dataflow\', \'process_delimited.py\')\n\n# The following additional Airflow variables should be set:\n# gcp_project:         Google Cloud Platform project id.\n# gcp_temp_location:   Google Cloud Storage location to use for Dataflow temp location.\nDEFAULT_DAG_ARGS = {\n    \'start_date\': YESTERDAY,\n    \'retries\': 2,\n\n    # TODO (done): Populate the models.Variable.get() with the variable name for your GCP Project\n    \'project_id\': models.Variable.get(\'gcp_project\'),  ## [[?]] or use a separate name, like gcp_project_id?\n    \'dataflow_default_options\': {\n        \'project\': models.Variable.get(\'gcp_project\'),\n\n        # TODO (done): Populate the models.Variable.get() with the variable name for temp location\n        \'temp_location\': \'gs://\'+models.Variable.get(\'gcp_temp_location\'),\n        \'runner\': \'DataflowRunner\'\n    }\n}\n\n\ndef move_to_completion_bucket(target_bucket, target_infix, **kwargs):\n    """A utility method to move an object to a target location in GCS."""\n    # Here we establish a connection hook to GoogleCloudStorage.\n    # Google Cloud Composer automatically provides a google_cloud_storage_default\n    # connection id that is used by this hook.\n    conn = gcs_hook.GoogleCloudStorageHook()\n\n    # The external trigger (Google Cloud Function) that initiates this DAG\n    # provides a dag_run.conf dictionary with event attributes that specify\n    # the information about the GCS object that triggered this DAG.\n    # We extract the bucket and object name from this dictionary.\n    source_bucket = models.Variable.get(\'gcp_input_location\')\n    source_object = models.Variable.get(\'gcp_input_location\')+\'/usa_names.csv\' \n    completion_ds = kwargs[\'ds\']\n\n    target_object = os.path.join(target_infix, completion_ds, source_object)\n\n    logging.info(\'Copying %s to %s\',\n                 os.path.join(source_bucket, source_object),\n                 os.path.join(target_bucket, target_object))\n    conn.copy(source_bucket, source_object, target_bucket, target_object)\n\n    logging.info(\'Deleting %s\',\n                 os.path.join(source_bucket, source_object))\n    conn.delete(source_bucket, source_object)\n\n\n# Setting schedule_interval to None as this DAG is externally trigger by a Cloud Function.\n# The following Airflow variables should be set for this DAG to function:\n# bq_output_table: BigQuery table that should be used as the target for\n#                  Dataflow in <dataset>.<tablename> format.\n#                  e.g. lake.usa_names\n# input_field_names: Comma separated field names for the delimited input file.\n#                  e.g. state,gender,year,name,number,created_date\n\n# TODO (done): Name the DAG id GcsToBigQueryTriggered\nwith models.DAG(dag_id=\'GcsToBigQueryTriggered\',\n                description=\'A DAG triggered by an external Cloud Function\',\n                schedule_interval=None, \n                default_args=DEFAULT_DAG_ARGS) as dag:\n    # Args required for the Dataflow job.\n    job_args = {\n        \'input\': INPUT_BUCKET_CSV,\n\n        # TODO (done): Populate the models.Variable.get() with the variable name for BQ table\n        \'output\': \'gs://\'+models.Variable.get(\'gcp_bq_target_table\'),\n\n        # TODO (done): Populate the models.Variable.get() with the variable name for input field names\n        \'fields\': models.Variable.get(\'gcp_input_field_names\'),\n        \'load_dt\': DS_TAG\n    }\n\n    # Main Dataflow task that will process and load the input delimited file.\n    # TODO (done): Specify the type of operator we need to call to invoke DataFlow\n    dataflow_task = dataflow_operator.DataFlowPythonOperator(\n        task_id="process-delimited-and-push",\n        py_file=DATAFLOW_FILE,\n        options=job_args)\n\n    # Here we create two conditional tasks, one of which will be executed\n    # based on whether the dataflow_task was a success or a failure.\n    success_move_task = python_operator.PythonOperator(task_id=\'success-move-to-completion\',\n                                                       python_callable=move_to_completion_bucket,\n                                                       # A success_tag is used to move\n                                                       # the input file to a success\n                                                       # prefixed folder.\n                                                       op_args=[models.Variable.get(\'gcp_completion_bucket\'), SUCCESS_TAG],\n                                                       provide_context=True,\n                                                       trigger_rule=TriggerRule.ALL_SUCCESS)\n\n    failure_move_task = python_operator.PythonOperator(task_id=\'failure-move-to-completion\',\n                                                       python_callable=move_to_completion_bucket,\n                                                       # A failure_tag is used to move\n                                                       # the input file to a failure\n                                                       # prefixed folder.\n                                                       op_args=[models.Variable.get(\'gcp_completion_bucket\'), FAILURE_TAG],\n                                                       provide_context=True,\n                                                       trigger_rule=TriggerRule.ALL_FAILED)\n\n    # The success_move_task and failure_move_task are both downstream from the\n    # dataflow_task.\n    dataflow_task >> success_move_task\n    dataflow_task >> failure_move_task')


# ## Viewing environment information
# Now that you have a completed DAG, it's time to copy it to your Cloud Composer environment and finish the setup of your workflow.<br><br>
# 1. Go back to **Composer** to check on the status of your environment.
# 2. Once your environment has been created, click the **name of the environment** to see its details.
# <br><br>
# The Environment details page provides information, such as the Airflow web UI URL, Google Kubernetes Engine cluster ID, name of the Cloud Storage bucket connected to the DAGs folder.
# <br><br>
# Cloud Composer uses Cloud Storage to store Apache Airflow DAGs, also known as workflows. Each environment has an associated Cloud Storage bucket. Cloud Composer schedules only the DAGs in the Cloud Storage bucket.

# ## Setting Airflow variables
# Our DAG relies on variables to pass in values like the GCP Project. We can set these in the Admin UI.
# 
# Airflow variables are an Airflow-specific concept that is distinct from [environment variables](https://cloud.google.com/composer/docs/how-to/managing/environment-variables). In this step, you'll set the following six [Airflow variables](https://airflow.apache.org/concepts.html#variables) used by the DAG we will deploy.

# In[7]:


get_ipython().run_cell_magic('bash', '', 'cat simple_load_dag.py | grep "Variable.get(\'"')


# In[8]:


## Run this to display which key value pairs to input
import pandas as pd
pd.DataFrame([
  ('gcp_project', PROJECT),
  ('gcp_input_location', PROJECT + '_input'),
  ('gcp_temp_location', PROJECT + '_output/tmp'),
  ('gcp_completion_bucket', PROJECT + '_output'),
  ('gcp_input_field_names', 'state,gender,year,name,number,created_date'),
  ('gcp_bq_target_table', 'ml_pipeline.ingest_table')
], columns = ['Key', 'Value'])


# ### Option 1: Set the variables using the Airflow webserver UI
# 1. In your Airflow environment, select **Admin** > **Variables**
# 2. Populate each key value in the table with the required variables from the above table

# ### Option 2: Set the variables using the Airflow CLI
# The next gcloud composer command executes the Airflow CLI sub-command [variables](https://airflow.apache.org/cli.html#variables). The sub-command passes the arguments to the gcloud command line tool.<br><br>
# To set the variables, run the gcloud composer command once for each row from the above table. For instance, to set `gcp_project`:

# In[9]:


get_ipython().run_cell_magic('bash', '', 'gcloud components install kubectl')


# In[13]:


get_ipython().run_cell_magic('bash', '', '# REPLACE ENVIRONMENT_NAME WITH YOUR COMPOSER NAME \n# gcloud composer environments run ENVIRONMENT_NAME \\\n#  --location ${REGION} variables -- \\\n#  --set gcp_project ${PROJECT}\n\ngcloud composer environments run mlcomposer \\\n --location ${REGION} variables -- \\\n --set gcp_project ${PROJECT}')


# ### Copy your Airflow bucket name
# 1. Navigate to your Cloud Composer [instance](https://console.cloud.google.com/composer/environments?project=)<br/><br/>
# 2. Select __DAGs Folder__<br/><br/>
# 3. You will be taken to the Google Cloud Storage bucket that Cloud Composer has created automatically for your Airflow instance<br/><br/>
# 4. __Copy the bucket name__ into the variable below (example: us-central1-composer-08f6edeb-bucket)

# In[14]:


AIRFLOW_BUCKET = 'europe-west1-mlcomposer-1301ad19-bucket' # REPLACE WITH AIRFLOW BUCKET NAME (done)
os.environ['AIRFLOW_BUCKET'] = AIRFLOW_BUCKET


# ### Copy your Airflow files to your Airflow bucket

# In[19]:


get_ipython().run_cell_magic('bash', '', 'gsutil cp simple_load_dag.py gs://${AIRFLOW_BUCKET}/dags # overwrite DAG file if it exists\ngsutil cp -r dataflow/process_delimited.py gs://${AIRFLOW_BUCKET}/dags # copy Dataflow job to be ran')


# In[18]:


get_ipython().run_cell_magic('bash', '', 'gsutil ls gs://${AIRFLOW_BUCKET}/dags')


# ***
# ## Navigating Using the Airflow UI
# To access the Airflow web interface using the GCP Console:
# 1. Go back to the **Composer Environments** page.
# 2. In the **Airflow webserver** column for the environment, click the new window icon. 
# 3. The Airflow web UI opens in a new browser window. 

# ### Trigger DAG run manually
# Running your DAG manually ensures that it operates successfully even in the absence of triggered events. 
# 1. Trigger the DAG manually **click the play button** under Links
# 

# ***
# # Part Two: Trigger DAG run automatically from a file upload to GCS
# Now that your manual workflow runs successfully, you will now trigger it based on an external event. 

# ## Create a Cloud Function to trigger your workflow
# We will be following this [reference guide](https://cloud.google.com/composer/docs/how-to/using/triggering-with-gcf) to setup our Cloud Function
# 1. In the code block below, change the project_id, location, and composer_environment and populate them
# 2. Run the below code to get your **CLIENT_ID** (needed later)

# In[ ]:


import google.auth
import google.auth.transport.requests
import requests
import six.moves.urllib.parse

# Authenticate with Google Cloud.
# See: https://cloud.google.com/docs/authentication/getting-started
credentials, _ = google.auth.default(
    scopes=['https://www.googleapis.com/auth/cloud-platform'])
authed_session = google.auth.transport.requests.AuthorizedSession(
    credentials)

project_id = 'your-project-id'
location = 'us-central1'
composer_environment = 'composer'

environment_url = (
    'https://composer.googleapis.com/v1beta1/projects/{}/locations/{}'
    '/environments/{}').format(project_id, location, composer_environment)
composer_response = authed_session.request('GET', environment_url)
environment_data = composer_response.json()
airflow_uri = environment_data['config']['airflowUri']

# The Composer environment response does not include the IAP client ID.
# Make a second, unauthenticated HTTP request to the web server to get the
# redirect URI.
redirect_response = requests.get(airflow_uri, allow_redirects=False)
redirect_location = redirect_response.headers['location']

# Extract the client_id query parameter from the redirect.
parsed = six.moves.urllib.parse.urlparse(redirect_location)
query_string = six.moves.urllib.parse.parse_qs(parsed.query)
print(query_string['client_id'][0])


# ## Create the Cloud Function
# 
# 1. Navigate to **Compute** > **Cloud Functions**
# 2. Select **Create function**
# 3. For name specify **'gcs-dag-trigger-function'**
# 4. For trigger type select **'Cloud Storage'**
# 5. For event type select **'Finalize/Create'**
# 6. For bucket, **specify the input bucket** you created earlier 
# 
# Important: be sure to select the input bucket and not the output bucket to avoid an endless triggering loop

# ### populate index.js
# Complete the four required constants defined below in index.js code and **paste it into the Cloud Function editor** (the js code will not run in this notebook). The constants are: 
# - PROJECT_ID
# - CLIENT_ID (from earlier)
# - WEBSERVER_ID (part of Airflow webserver URL) 
# - DAG_NAME (GcsToBigQueryTriggered)

# In[ ]:


'use strict';

const fetch = require('node-fetch');
const FormData = require('form-data');

**()
 * Triggered from a message on a Cloud Storage bucket.
 *
 * IAP authorization based on:
 * https://stackoverflow.com/questions/45787676/how-to-authenticate-google-cloud-functions-for-access-to-secure-app-engine-endpo
 * and
 * https://cloud.google.com/iap/docs/authentication-howto
 *
 * @param {!Object} event The Cloud Functions event.
 * @param {!Function} callback The callback function.
 */
exports.triggerDag = function triggerDag (event, callback) {
  // Fill in your Composer environment information here.

  // The project that holds your function
  const PROJECT_ID = 'your-project-id'; 
  // example: qwiklabs-gcp-97d55fb651b04b20

  // Navigate to your webserver's login page and get this from the URL
  const CLIENT_ID = '';
  (/, example:, 954510698485-gde6id87qtdn9itl7809uj8s6a60n9gl)

  (/, This, should, be, part, of, your, webserver's, URL:)
  (/, {tenant-project-id}.appspot.com)
  const WEBSERVER_ID = '';
  (/, example:, b93193d731fd74d3f-tp)

  (/, The, name, of, the, DAG, you, wish, to, trigger)
  const DAG_NAME = 'GcsToBigQueryTriggered';
  (/, example:, GcsToBigQueryTriggered)

  (//////////////////////)
  (/, DO, NOT, EDIT, BELOW, //)

  (/, Other, constants)
  const WEBSERVER_URL = `https://${WEBSERVER_ID}.appspot.com/api/experimental/dags/${DAG_NAME}/dag_runs`;
  const USER_AGENT = 'gcf-event-trigger';
  const BODY = {'conf': JSON.stringify(event.data)};

  (/, Make, the, request)
  authorizeIap(CLIENT_ID, PROJECT_ID, USER_AGENT)
    .then(function iapAuthorizationCallback (iap) {
      makeIapPostRequest(WEBSERVER_URL, BODY, iap.idToken, USER_AGENT, iap.jwt);
    })
    .then(_ => callback(null))
    .catch(callback);
};

**()
   * @param {string} clientId The client id associated with the Composer webserver application.
   * @param {string} projectId The id for the project containing the Cloud Function.
   * @param {string} userAgent The user agent string which will be provided with the webserver request.
   */
function authorizeIap (clientId, projectId, userAgent) {
  const SERVICE_ACCOUNT = `${projectId}@appspot.gserviceaccount.com`;
  const JWT_HEADER = Buffer.from(JSON.stringify({alg: 'RS256', typ: 'JWT'}))
    .toString('base64');

  var jwt = '';
  var jwtClaimset = '';

  (/, Obtain, an, Oauth2, access, token, for, the, appspot, service, account)
  return fetch(
    `http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/${SERVICE_ACCOUNT}/token`,
    {
      headers: {'User-Agent': userAgent, 'Metadata-Flavor': 'Google'}
    })
    .then(res => res.json())
    .then(function obtainAccessTokenCallback (tokenResponse) {
      if (tokenResponse.error) {
        return Promise.reject(tokenResponse.error);
      }
      var accessToken = tokenResponse.access_token;
      var iat = Math.floor(new Date().getTime() / 1000);
      var claims = {
        iss: SERVICE_ACCOUNT,
        aud: 'https://www.googleapis.com/oauth2/v4/token',
        iat: iat,
        exp: iat + 60,
        target_audience: clientId
      };
      jwtClaimset = Buffer.from(JSON.stringify(claims)).toString('base64');
      var toSign = [JWT_HEADER, jwtClaimset].join('.');

      return fetch(
        `https://iam.googleapis.com/v1/projects/${projectId}/serviceAccounts/${SERVICE_ACCOUNT}:signBlob`,
        {
          method: 'POST',
          body: JSON.stringify({'bytesToSign': Buffer.from(toSign).toString('base64')}),
          headers: {
            'User-Agent': userAgent,
            'Authorization': `Bearer ${accessToken}`
          }
        });
    })
    .then(res => res.json())
    .then(function signJsonClaimCallback (body) {
      if (body.error) {
        return Promise.reject(body.error);
      }
      // Request service account signature on header and claimset
      var jwtSignature = body.signature;
      jwt = [JWT_HEADER, jwtClaimset, jwtSignature].join('.');
      var form = new FormData();
      form.append('grant_type', 'urn:ietf:params:oauth:grant-type:jwt-bearer');
      form.append('assertion', jwt);
      return fetch(
        'https://www.googleapis.com/oauth2/v4/token', {
          method: 'POST',
          body: form
        });
    })
    .then(res => res.json())
    .then(function returnJwt (body) {
      if (body.error) {
        return Promise.reject(body.error);
      }
      return {
        jwt: jwt,
        idToken: body.id_token
      };
    });
}

**()
   * @param {string} url The url that the post request targets.
   * @param {string} body The body of the post request.
   * @param {string} idToken Bearer token used to authorize the iap request.
   * @param {string} userAgent The user agent to identify the requester.
   * @param {string} jwt A Json web token used to authenticate the request.
   */
function makeIapPostRequest (url, body, idToken, userAgent, jwt) {
  var form = new FormData();
  form.append('grant_type', 'urn:ietf:params:oauth:grant-type:jwt-bearer');
  form.append('assertion', jwt);

  return fetch(
    url, {
      method: 'POST',
      body: form
    })
    .then(function makeIapPostRequestCallback () {
      return fetch(url, {
        method: 'POST',
        headers: {
          'User-Agent': userAgent,
          'Authorization': `Bearer ${idToken}`
        },
        body: JSON.stringify(body)
      });
    });
}


# ### populate package.json
# Copy and paste the below into **package.json**

# In[ ]:


{
  "name": "nodejs-docs-samples-functions-composer-storage-trigger",
  "version": "0.0.1",
  "dependencies": {
    "form-data": "^2.3.2",
    "node-fetch": "^2.2.0"
  },
  "engines": {
    "node": ">=4.3.2"
  },
  "private": true,
  "license": "Apache-2.0",
  "author": "Google Inc.",
  "repository": {
    "type": "git",
    "url": "https://github.com/GoogleCloudPlatform/nodejs-docs-samples.git"
  },
  "devDependencies": {
    "@google-cloud/nodejs-repo-tools": "^2.2.5",
    "ava": "0.25.0",
    "proxyquire": "2.0.0",
    "semistandard": "^12.0.1",
    "sinon": "4.4.2"
  },
  "scripts": {
    "lint": "repo-tools lint",
    "test": "ava -T 20s --verbose test/*.test.js"
  }
}


# 7. For **Function to execute**, specify **triggerDag** (note: case sensitive)
# 8. Select **Create**

# ## Upload CSVs and Monitor
# 1. Practice uploading and editing CSVs named usa_names.csv into your input bucket (note: the DAG filters to only ingest CSVs with 'usa_names.csv' as the filepath. Adjust this as needed in the DAG code.)
# 2. Troubleshoot Cloud Function call errors by monitoring the [logs](https://console.cloud.google.com/logs/viewer?). In the below screenshot we filter in Logging for our most recent Dataflow job and are scrolling through to ensure the job is processing and outputting records to BigQuery
# 
# ![Dataflow logging](./img/dataflow_logging.jpg "Dataflow logging")
# 
# 3. Troubleshoot Airflow workflow errors by monitoring the **Browse** > **DAG Runs** 

# ## Congratulations! 
# You’ve have completed this advanced lab on triggering a workflow with a Cloud Function.