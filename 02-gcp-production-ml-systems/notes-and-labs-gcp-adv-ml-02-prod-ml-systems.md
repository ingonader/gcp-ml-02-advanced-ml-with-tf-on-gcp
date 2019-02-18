# GCP Production ML Systems



## to download materials

software to download materials: <https://github.com/coursera-dl/coursera-dl>

link to coursera material: <https://www.coursera.org/learn/end-to-end-ml-tensorflow-gcp/lecture/wWBkE/effective-ml>

```bash
coursera-dl -u "ingo.nader@gmail.com" gcp-production-ml-systems --download-quizzes --download-notebooks --about
```



## Notes

* Do labs in Google Chrome (recommended)
* Open Incognito window, go to <http://console.cloud.google.com/>



# Lab 1: [ML on GCP C7] Serving on Cloud MLE

## Overview

*Duration is 1 min*

In this lab, you build an AppEngine app to serve your ML predictions. You get to modify the user-facing form and the python script and deploy them as an AppEngine app that makes requests to your deployed ML model.

### **What you learn**

In this lab, you:

* modify a simple UI form to get user input when making calls to your model
* build the http request to be made to your deployed ML model on Cloud MLE

## Setup

For each lab, you get a new GCP project and set of resources for a fixed time at no cost.

1. Make sure you signed into Qwiklabs using an **incognito window**.
2. Note the lab's access time (for example, ![img/time.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/699409e014fbb8298cf5747a0535e04d13f51dfbb54fbaba57e7276dd12c6e95.png) and make sure you can finish in that time block.

There is no pause feature. You can restart if needed, but you have to start at the beginning.

1. When ready, click ![img/start_lab.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/5c4f31eeebd0a24cae6cdc2760a29cfaf04136b325a3988d6620c3cd04370dda.png).
2. Note your lab credentials. You will use them to sign in to Cloud Platform Console. ![img/open_console.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/5eaee037e6cedbf49f6a702ab8a9ef820bb8e717332946ff76a0831a6396aafc.png)
3. Click **Open Google Console**.
4. Click **Use another account** and copy/paste credentials for **this** lab into the prompts.

If you use other credentials, you'll get errors or **incur charges**.

1. Accept the terms and skip the recovery resource page.

Do not click **End** unless you are finished with the lab or want to restart it. This clears your work and removes the project.

## Start Cloud Shell

### Activate Google Cloud Shell

Google Cloud Shell provides command-line access to your GCP resources.

From the GCP Console click the **Cloud Shell** icon on the top right toolbar:

![Cloud Shell Icon](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/718029dee0e562c61c14536c5a636a5bae0ef5136e9863b98160d1e06123908a.png)

Then click **START CLOUD SHELL**:

![Start Cloud Shell](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/feb5ea74b4a4f6dfac7800f39c3550364ed7a33a7ab17b6eb47cab3e65c33b13.png)

You can click **START CLOUD SHELL** immediately when the dialog comes up instead of waiting in the dialog until the Cloud Shell provisions.

It takes a few moments to provision and connects to the environment:

![Cloud Shell Terminal](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/11def2e8f4cfd6f1590f3fd825d4566658501ca87e1d5d1552aa17339050c194.png)

The Cloud Shell is a virtual machine loaded with all the development tools you’ll need. It offers a persistent 5GB home directory, and runs on the Google Cloud, greatly enhancing network performance and authentication.

Once connected to the cloud shell, you'll see that you are already authenticated and the project is set to your *PROJECT_ID*:

```
gcloud auth list
```

Output:

```output
Credentialed accounts:
 - <myaccount>@<mydomain>.com (active)
```

**Note:** `gcloud` is the powerful and unified command-line tool for Google Cloud Platform. Full documentation is available on [Google Cloud gcloud Overview](https://cloud.google.com/sdk/gcloud). It comes pre-installed on Cloud Shell and supports tab-completion.

```
gcloud config list project
```

Output:

```output
[core]
project = <PROJECT_ID>
```

## Copy trained model

### **Step 1**

Set necessary variables and create a bucket:

```bash
REGION=us-central1
BUCKET=$(gcloud config get-value project)
TFVERSION=1.7
gsutil mb -l ${REGION} gs://${BUCKET}
```

### **Step 2**

Copy trained model into your bucket:

```bash
gsutil -m cp -R gs://cloud-training-demos/babyweight/trained_model gs://${BUCKET}/babyweight
```

## Deploy trained model

### **Step 1**

Set necessary variables:

```bash
MODEL_NAME=babyweight
MODEL_VERSION=ml_on_gcp
MODEL_LOCATION=$(gsutil ls gs://${BUCKET}/babyweight/export/exporter/ | tail -1)
```

### **Step 2**

Deploy trained model:

```bash
gcloud ml-engine models create ${MODEL_NAME} --regions $REGION
gcloud ml-engine versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version $TFVERSION
```

## Code for your frontend

### **Step 1**

Clone the course repository:

```bash
cd ~
git clone https://github.com/GoogleCloudPlatform/training-data-analyst
```

### **Step 2**

You can use the Cloud Shell code editor to view and edit the contents of these files.

Click on the (![b8ebde10ba2a31c8.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/5cd735e9c729fee5765a0d6cf933bcd0357f5ebc6faa899bb26b8f3d740bbe2a.png)) icon on the top right of your Cloud Shell window to launch Code Editor.

Once launched, navigate to the `~/training-data-analyst/courses/machine_learning/deepdive/06_structured/labs/serving`directory.

### **Step 3**

Open the **application/main.py**and **application/templates/form.html** files and notice the *#TODO*s within the code. These need to be replaced with code. The next section tells you how.

## Modify main.py

### **Step 1**

Open the `main.py` file by clicking on it. Notice the lines with *# TODO* for setting credentials and the api to use.

Set the credentials to use Google Application Default Credentials (recommended way to authorize calls to our APIs when building apps deployed on AppEngine):

```bash
credentials = GoogleCredentials.get_application_default()
```

Specify the api name (ML Engine API) and version to use:

```bash
api = discovery.build('ml', 'v1', credentials=credentials)
```

### **Step 2**

Scroll further down in `main.py` and look for the next *#TODO* in the method `get_prediction()`. In there, specify, using the **parent** variable, the name of your trained model deployed on Cloud MLE:

```bash
parent = 'projects/%s/models/%s' % (project, model_name)
```

### **Step 3**

Now that you have all the pieces for making the call to your model, build the call request by specifying it in the **prediction** variable:

```bash
prediction = api.projects().predict(body=input_data, name=parent).execute()
```

### **Step 4**

The final *#TODO* (scroll towards bottom) is to get gestation_weeks from the form data and cast into a float within the **features** array:

```bash
features['gestation_weeks'] = float(data['gestation_weeks'])
```

### **Step 5**

Save the changes you made using the **File** > **Save** button on the top left of your code editor window.

![3b0e6c092072fec5.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/9cff32861ee2e2dfa534f609fa4acbf7418e3b3b9475dcb188cff5510cfc084b.png)

## Modify form.html

`form.html` is the front-end of your app. The user fills in data (features) about the mother based on which we will make the predictions using our trained model.

### **Step 1**

In code editor, navigate to the `application/templates` directory and click to open the `form.html` file.

### **Step 2**

There is one *#TODO* item here. Look for the div segment for **Plurality** and add options for other plurality values (2, 3, etc).

```bash
<md-option value="2">Twins</md-option>
<md-option value="3">Triplets</md-option>
```

### **Step 3**

Save the changes you made using the **File** > **Save** button on the top left of your code editor window.

## Deploy and test your app

### **Step 1**

In Cloud Shell, run the `deploy.sh` script to install required dependencies and deploy your app engine app to the cloud.

```bash
cd training-data-analyst/courses/machine_learning/deepdive/06_structured/labs/serving
./deploy.sh
```

Note: Choose a region for App Engine when prompted and follow the prompts during this process

### **Step 2**

Go to the url `https://<PROJECT-ID>.appspot.com` and start making predictions.

*Note: Replace <PROJECT-ID> with your Project ID.*

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.



# [[eof]]



[[todo]]:

* make python and html files of all notebooks in all courses!
* make pdfs of all .md-files in all courses! (add to .gitignore)