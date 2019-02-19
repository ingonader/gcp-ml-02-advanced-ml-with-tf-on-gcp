# GCP Production ML Systems



## to download materials

software to download materials: <https://github.com/coursera-dl/coursera-dl>

link to coursera material: <https://www.coursera.org/learn/end-to-end-ml-tensorflow-gcp/lecture/wWBkE/effective-ml>

```bash
coursera-dl -u "ingo.nader@gmail.com" gcp-production-ml-systems -sl en --download-quizzes --download-notebooks --about

#coursera-dl -u "ingo.nader@gmail.com" gcp-production-ml-systems -sl "en,de" --download-quizzes --download-notebooks --about
```

Done with modified version of coursera-dl.

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



# Lab 2: [ML on GCP C7] Serving ML Predictions in batch and real-time

## Overview

*Duration is 1 min*

In this lab, you run Dataflow pipelines to serve predictions for batch requests as well as streaming in real-time.

### **What you learn**

In this lab, you write code to:

* Create a prediction service that calls your trained model deployed in Cloud to serve predictions
* Run a Dataflow job to have the prediction service read in batches from a CSV file and serve predictions
* Run a streaming Dataflow pipeline to read requests real-time from Cloud Pub/Sub and write predictions into a BigQuery table

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

## Browse lab files

*Duration is 5 min*

### **Step 1**

Clone the course repository:

```bash
cd ~
#git clone https://github.com/GoogleCloudPlatform/training-data-analyst

export GITREPO=gcp-ml-02-advanced-ml-with-tf-on-gcp
#cd /content/datalab/
git clone https://github.com/ingonader/${GITREPO}.git
cd $GITREPO
git config user.email "ingo.nader@gmail.com"
git config user.name "Ingo Nader"


```

### **Step 2**

In Cloud Shell, navigate to the folder containing the code for this lab:

```bash
#cd ~/training-data-analyst/courses/machine_learning/deepdive/06_structured/labs/serving
cd ./02-gcp-production-ml-systems/labs/serving


```

### **Step 3**

Run the `what_to_fix.sh` script to see a list of items you need to add/modify to existing code to run your app:

```bash
./what_to_fix.sh
```

As a result of this, you will see a list of filenames and lines within those files marked with **TODO**. These are the lines where you have to add/modify code. For this lab, you will focus on #TODO items for **.java files only**, namely `BabyweightMLService.java` : which is your prediction service.

## How the code is organized

![f2aeed7941e38072.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/98e07b83f032c6704f21c98e42b9ed26d2c99cba4fb3554dc1e4c46a1607c3be.png)

## Prediction service

In this section, you fix the code in **BabyweightMLService.java** and test it with the **run_once.sh** script that is provided. If you need help with the code, look at the next section that provides hints on how to fix code in BabyweightMLService.java.

### **Step 1**

You may use the Cloud Shell code editor to view and edit the contents of these files.

Click on the (![b8ebde10ba2a31c8.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/5cd735e9c729fee5765a0d6cf933bcd0357f5ebc6faa899bb26b8f3d740bbe2a.png)) icon on the top right of your Cloud Shell window to launch Code Editor.

### **Step 2**

After it is launched, navigate to the following directory: `training-data-analyst/courses/machine_learning/deepdive/06_structured/labs/serving/pipeline/src/main/java/com/google/cloud/training/mlongcp`

### **Step 3**

Open the `BabyweightMLService.java` files and replace *#TODOs* in the code.

### **Step 4**

Once completed, go into your Cloud Shell and run the `run_once.sh`script to test your ML service.

```bash
cd ~/training-data-analyst/courses/machine_learning/deepdive/06_structured/labs/serving
./run_once.sh
## <... lots of maven output ...>
## result:
## predicted=6.9363484382629395 actual=5.423372
## [INFO] ------------------------------------------------------------------------
## [INFO] BUILD SUCCESS
## [INFO] ------------------------------------------------------------------------
## [INFO] Total time:  01:02 min
## [INFO] Finished at: 2019-02-18T15:28:09+01:00
## [INFO] ------------------------------------------------------------------------
```

**`run_once.sh` Listing:**

```bash
#!/bin/bash

cd pipeline
rm -rf ../output
mvn compile exec:java \
 -Dexec.mainClass=com.google.cloud.training.mlongcp.BabyweightMLService
```



## Serve predictions for batch requests

This section of the lab calls AddPrediction.java that takes a batch input (one big CSV), calls the prediction service to generate baby weight predictions and writes them into local files (multiple CSVs).

### **Step 1**

In your Cloud Shell code editor, open the `AddPrediction.java` file available in the following directory: `training-data-analyst/courses/machine_learning/deepdive/06_structured/labs/serving/pipeline/src/main/java/com/google/cloud/training/mlongcp`


### **Step 2**

Look through the code and notice how, based on input argument, it decides to set up a batch or streaming pipeline, and creates the appropriate TextInputOutput or PubSubBigQuery io object respectively to handle the reading and writing.

**Note:** Look back at the diagram in "how code is organized" section to make sense of it all.

```java
  public static void main(String[] args) {
    MyOptions options = PipelineOptionsFactory.fromArgs(args).withValidation().as(MyOptions.class);
    if (options.isRealtime()) {
      options.setStreaming(true);
      options.setRunner(DataflowRunner.class);
    }    
    
   
    Pipeline p = Pipeline.create(options);

    InputOutput io;
    if (options.isRealtime()) {
      // in real-time, we read from PubSub and write to BigQuery
      io = new PubSubBigQuery();
    } else {
      io = new TextInputOutput();
    } 
    
    PCollection<Baby> babies = io //
        .readInstances(p, options) //
        .apply(Window.<Baby> into(FixedWindows.of(Duration.standardSeconds(20))).withAllowedLateness(Duration.standardSeconds(10)).discardingFiredPanes());

    io.writePredictions(babies, options);

    PipelineResult result = p.run();
    if (!options.isRealtime()) {
      result.waitUntilFinish();
    }
  }
}

```

### **Step 3**

Test batch mode by running the `run_ontext.sh` script provided in the lab directory:

```bash
cd ~/training-data-analyst/courses/machine_learning/deepdive/06_structured/labs/serving
./run_ontext.sh
# Your active configuration is: [cloudshell-32200]
# [INFO] Scanning for projects...
# [INFO]
# [INFO] -------< com.google.cloud.training.mlongcp.babyweight:pipeline >--------
# [INFO] Building pipeline [1.0.0,2.0.0]
# [INFO] --------------------------------[ jar ]---------------------------------
# [INFO]
# [INFO] --- maven-resources-plugin:2.6:resources (default-resources) @ pipeline ---
# [INFO] Using 'UTF-8' encoding to copy filtered resources.
# [INFO] skip non existing resourceDirectory /home/google2479254_student/gcp-ml-02-advanced-ml-with-tf-on-gcp/02-gcp-production-ml-systems/labs/serving/pipeline/src/main/resources
# [INFO]
# [INFO] --- maven-compiler-plugin:3.5.1:compile (default-compile) @ pipeline ---
# [INFO] Changes detected - recompiling the module!
# [INFO] Compiling 7 source files to /home/google2479254_student/gcp-ml-02-advanced-ml-with-tf-on-gcp/02-gcp-production-ml-systems/labs/serving/pipeline/target/classes
# [INFO]
# [INFO] --- exec-maven-plugin:1.4.0:java (default-cli) @ pipeline ---
# Feb 18, 2019 3:32:44 PM com.google.cloud.training.mlongcp.TextInputOutput readInstances
# INFO: Reading data from ../*.csv.gz
# Feb 18, 2019 3:32:47 PM org.apache.beam.sdk.io.FileBasedSource getEstimatedSizeBytes
# INFO: Filepattern ../*.csv.gz matched 1 files with total size 504345
# Feb 18, 2019 3:32:47 PM org.apache.beam.sdk.io.FileBasedSource split
# INFO: Splitting filepattern ../*.csv.gz into bundles of size 168115 took 19 ms and produced 1 files and 1 bundles
# Feb 18, 2019 3:33:21 PM com.google.cloud.training.mlongcp.BabyweightMLService mock_batchPredict
# INFO: Mock prediction for 29898 instances
# Feb 18, 2019 3:33:21 PM com.google.cloud.training.mlongcp.BabyweightMLService mock_batchPredict
# INFO: Mock prediction for 30082 instances
# Feb 18, 2019 3:33:40 PM org.apache.beam.sdk.io.WriteFiles$WriteShardsIntoTempFilesFn processElement
# INFO: Opening writer 08b231de-1456-463b-8324-da3f9aa274f7 for window org.apache.beam.sdk.transforms.windowing.GlobalWindow@6755a5bf pane PaneInfo{isFirst=true, isLast=true, timing=ON_TIME, index=0, onTimeIndex=0} destination null
# Feb 18, 2019 3:33:40 PM org.apache.beam.sdk.io.WriteFiles$WriteShardsIntoTempFilesFn processElement
# INFO: Opening writer 9e86b208-57fc-43ac-84c6-29fcdf4a2804 for window org.apache.beam.sdk.transforms.windowing.GlobalWindow@6755a5bf pane PaneInfo{isFirst=true, isLast=true, timing=ON_TIME, index=0, onTimeIndex=0} destination null
# Feb 18, 2019 3:33:40 PM org.apache.beam.sdk.io.WriteFiles$WriteShardsIntoTempFilesFn processElement
# Feb 18, 2019 3:33:41 PM org.apache.beam.sdk.io.WriteFiles$WriteShardsIntoTempFilesFn processElement
# INFO: Opening writer 2b5e03f3-029f-427a-b7fc-aac037add8b0 for window org.apache.beam.sdk.transforms.windowing.GlobalWindow@6755a5bf pane PaneInfo{isFirst=true, isLast=true, timing=ON_TIME, index=0, onTimeIndex=0} destination null
# Feb 18, 2019 3:33:41 PM org.apache.beam.sdk.io.WriteFiles$WriteShardsIntoTempFilesFn processElement
# INFO: Opening writer 04d65f75-6aef-4806-be85-ae54cf8be606 for window org.apache.beam.sdk.transforms.windowing.GlobalWindow@6755a5bf pane PaneInfo{isFirst=true, isLast=true, timing=ON_TIME, index=0, onTimeIndex=0} destination null
# Feb 18, 2019 3:33:41 PM org.apache.beam.sdk.io.FileBasedSink$Writer close
# INFO: Successfully wrote temporary file /home/google2479254_student/gcp-ml-02-advanced-ml-with-tf-on-gcp/02-gcp-production-ml-systems/labs/serving/output/.temp-beam-2019-02-18_14-32-45-1/2b5e03f3-029f-427a-b7fc-aac037add8b0
# Feb 18, 2019 3:33:41 PM org.apache.beam.sdk.io.FileBasedSink$Writer close
# INFO: Successfully wrote temporary file /home/google2479254_student/gcp-ml-02-advanced-ml-with-tf-on-gcp/02-gcp-production-ml-systems/labs/serving/output/.temp-beam-2019-02-18_14-32-45-1/04d65f75-6aef-4806-be85-ae54cf8be606
# Feb 18, 2019 3:33:41 PM org.apache.beam.sdk.io.WriteFiles$FinalizeTempFileBundles$FinalizeFn process
# INFO: Finalizing 5 file results
# Feb 18, 2019 3:33:41 PM org.apache.beam.sdk.io.FileBasedSink$WriteOperation createMissingEmptyShards
# INFO: Finalizing for destination null num shards 5.
# Feb 18, 2019 3:33:41 PM org.apache.beam.sdk.io.FileBasedSink$WriteOperation moveToOutputFiles
# INFO: Will copy temporary file FileResult{tempFilename=/home/google2479254_student/gcp-ml-02-advanced-ml-with-tf-on-gcp/02-gcp-production-ml-systems/labs/serving/output/.temp-beam-2019-02-18_14-32-45-1/08b231de-1456-463b-8324-da3f9aa274f7, shard=0, window=org.apache.beam.sdk.transforms.windowing.GlobalWindow@6755a5bf, paneInfo=PaneInfo{isFirst=true, isLast=true, timing=ON_TIME, index=0, onTimeIndex=0}} to final location /home/google2479254_student/gcp-ml-02-advanced-ml-with-tf-on-gcp/02-gcp-production-ml-systems/labs/serving/output/flightPreds-00000-of-00005.csv
# Feb 18, 2019 3:33:41 PM org.apache.beam.sdk.io.FileBasedSink$WriteOperation moveToOutputFiles
# INFO: Will copy temporary file FileResult{tempFilename=/home/google2479254_student/gcp-ml-02-advanced-ml-with-tf-on-gcp/02-gcp-production-ml-systems/labs/serving/output/.temp-beam-2019-02-18_14-32-45-1/7a0eaf46-6882-4176-8343-f8178e876532, shard=1, window=org.apache.beam.sdk.transforms.windowing.GlobalWindow@6755a5bf, paneInfo=PaneInfo{isFirst=true, isLast=true, timing=ON_TIME, index=0, onTimeIndex=0}} to final location /home/google2479254_student/gcp-ml-02-advanced-ml-with-tf-on-gcp/02-gcp-production-ml-systems/labs/serving/output/flightPreds-00001-of-00005.csv
# Feb 18, 2019 3:33:41 PM org.apache.beam.sdk.io.FileBasedSink$WriteOperation moveToOutputFiles
# INFO: Will copy temporary file FileResult{tempFilename=/home/google2479254_student/gcp-ml-02-advanced-ml-with-tf-on-gcp/02-gcp-production-ml-systems/labs/serving/output/.temp-beam-2019-02-18_14-32-45-1/9e86b208-57fc-43ac-84c6-29fcdf4a2804, shard=2, window=org.apache.beam.sdk.transforms.windowing.GlobalWindow@6755a5bf, paneInfo=PaneInfo{isFirst=true, isLast=true, timing=ON_TIME, index=0, onTimeIndex=0}} to final location /home/google2479254_student/gcp-ml-02-advanced-ml-with-tf-on-gcp/02-gcp-production-ml-systems/labs/serving/output/flightPreds-00002-of-00005.csv
# Feb 18, 2019 3:33:41 PM org.apache.beam.sdk.io.FileBasedSink$WriteOperation moveToOutputFiles
# INFO: Will copy temporary file FileResult{tempFilename=/home/google2479254_student/gcp-ml-02-advanced-ml-with-tf-on-gcp/02-gcp-production-ml-systems/labs/serving/output/.temp-beam-2019-02-18_14-32-45-1/2b5e03f3-029f-427a-b7fc-aac037add8b0, shard=3, window=org.apache.beam.sdk.transforms.windowing.GlobalWindow@6755a5bf, paneInfo=PaneInfo{isFirst=true, isLast=true, timing=ON_TIME, index=0, onTimeIndex=0}} to final location /home/google2479254_student/gcp-ml-02-advanced-ml-with-tf-on-gcp/02-gcp-production-ml-systems/labs/serving/output/flightPreds-00003-of-00005.csv
# Feb 18, 2019 3:33:41 PM org.apache.beam.sdk.io.FileBasedSink$WriteOperation moveToOutputFiles
# INFO: Will copy temporary file FileResult{tempFilename=/home/google2479254_student/gcp-ml-02-advanced-ml-with-tf-on-gcp/02-gcp-production-ml-systems/labs/serving/output/.temp-beam-2019-02-18_14-32-45-1/04d65f75-6aef-4806-be85-ae54cf8be606, shard=4, window=org.apache.beam.sdk.transforms.windowing.GlobalWindow@6755a5bf, paneInfo=PaneInfo{isFirst=true, isLast=true, timing=ON_TIME, index=0, onTimeIndex=0}} to final location /home/google2479254_student/gcp-ml-02-advanced-ml-with-tf-on-gcp/02-gcp-production-ml-systems/labs/serving/output/flightPreds-00004-of-00005.csv
# Feb 18, 2019 3:33:41 PM org.apache.beam.sdk.io.FileBasedSink$WriteOperation removeTemporaryFiles
# INFO: Will remove known temporary file /home/google2479254_student/gcp-ml-02-advanced-ml-with-tf-on-gcp/02-gcp-production-ml-systems/labs/serving/output/.temp-beam-2019-02-18_14-32-45-1/04d65f75-6aef-4806-be85-ae54cf8be606
# <...>
# [INFO] ------------------------------------------------------------------------
# [INFO] BUILD SUCCESS
# [INFO] ------------------------------------------------------------------------
# [INFO] Total time:  01:09 min
# [INFO] Finished at: 2019-02-18T15:33:41+01:00
# [INFO] ------------------------------------------------------------------------
```

**`run_ontext.sh` Listing:**

```bash
#!/bin/bash

PROJECTID=$(gcloud config get-value project)

cd pipeline
rm -rf ../output
mvn compile exec:java \
 -Dexec.mainClass=com.google.cloud.training.mlongcp.AddPrediction \
 -Dexec.args="--input=../*.csv.gz --output=../output/ --project=$PROJECTID"
```



## Serve predictions real-time with a streaming pipeline

In this section of the lab, you will launch a streaming pipeline with Dataflow, which will accept incoming information from Cloud Pub/Sub, use the info to call the prediction service to get baby weight predictions, and finally write that info into a BigQuery table.

### **Step 1**

On your GCP Console's left-side menu, go into **Pub/Sub** and click the **CREATE TOPIC** button on top. Create a topic called **babies**.

![c4128dad787aaada.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/6f91cd2e14b3f6c050b5b441d6aebf22ef107287d8fc6c5aaf8e162aab3ef865.png)

### **Step 2**

Back in your Cloud Shell, modify the script __`run_dataflow.sh`__ to get Project ID (using *--project*) from command line arguments, and then run as follows:

```bash
cd ~/training-data-analyst/courses/machine_learning/deepdive/06_structured/labs/serving
./run_dataflow.sh
```

This will create a streaming Dataflow pipeline.

**Listing**:

```bash
#!/bin/bash

PROJECTID=$(gcloud config get-value project)

cd pipeline
bq mk babyweight
bq rm -rf babyweight.predictions

mvn compile exec:java \
 -Dexec.mainClass=com.google.cloud.training.mlongcp.AddPrediction \
 -Dexec.args="--realtime --input=babies --output=babyweight.predictions --project=$PROJECTID"
```



### **Step 3**

Back in your GCP Console, use the left-side menu to go into **Dataflow** and verify that the streaming job is created.

![eaf7891a8d680d8e.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/5b7d70aa0b60510f17a1a4c6ae1946d8d7c6f0156eeee8f76ec0e63165aa50f9.png)

### **Step 4**

Next, click on the job name to view the pipeline graph. Click on the pipeline steps (boxes) and look at the run details (like system lag, elements added, etc.) of that step on the right side.

![662fb484741d22e2.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/9c23617ee12b9bf11cb162077456a453604588cc27fe5ff9df024a207f4e4410.png)

This means that your pipeline is running and waiting for input. Let's provide input through the Pub/Sub topic.

### **Step 5**

Copy some lines from your example.csv.gz:

```bash
cd ~/training-data-analyst/courses/machine_learning/deepdive/06_structured/labs/serving
zcat exampledata.csv.gz
```

```
7.70295143428,Unknown,34,Single(1),39.0,-328012383083104805
8.12623897732,Unknown,34,Single(1),40.0,-328012383083104805
7.8484565272,Unknown,34,Single(1),40.0,-328012383083104805
7.8484565272,Unknown,34,Single(1),38.0,-328012383083104805
6.1244416383599996,Unknown,34,Single(1),40.0,-1525201076796226340
8.2342654857,Unknown,34,Single(1),39.0,-1525201076796226340
4.7509617461,Unknown,34,Multiple(2+),37.0,-1525201076796226340
8.8625829324,Unknown,34,Single(1),40.0,-1525201076796226340
9.06320359082,Unknown,34,Single(1),40.0,-1525201076796226340
8.08655577016,Unknown,34,Single(1),38.0,-1525201076796226340
8.18796841068,Unknown,34,Single(1),40.0,-1525201076796226340
6.7020527647999995,Unknown,34,Single(1),39.0,-1525201076796226340
8.062304921339999,Unknown,34,Single(1),40.0,-1525201076796226340
7.7492485093,Unknown,34,Single(1),38.0,-1525201076796226340
8.1460805809,Unknown,34,Single(1),39.0,-1525201076796226340
7.12534030784,Unknown,34,Single(1),41.0,-5937540421097454372
9.56365292556,Unknown,34,Single(1),40.0,-5937540421097454372
```



### **Step 6**

On your GCP Console, go back into **Pub/Sub**, click on the **babies** topic, and then click on **Publish message** button on top. In the message box, paste the lines you just copied from exampledata.csv.gz and click on **Publish** button.

![9e58fd14886fba1f.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/e7cdb22a26aee47f6da172f63c36be5e60e04b88f47ff19b788b4bb9e981bff0.png)

### **Step 7**

You may go back into Dataflow jobs on your GCP Console, click on your job and see how the run details have changed for the steps, for example click on write_toBQ and look at Elements added.

### **Step 8**

Lets verify that the predicted weights have been recorded into the BigQuery table.

1. Open the BigQuery ‘Classic’ console UI in a new tab: [BigQuery](https://bigquery.cloud.google.com/)

You may be prompted to enter your lab account’s password again.

**Note**: Another way to open BigQuery is to open the navigation menu (![HorizontalLine](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/b64830d530e08f843e60a4145b88d415dd0ee4e10a9543014586e1942ac5d166.png)) in the GCP console and then click the BigQuery link. This will open the new BigQuery UI, which is currently in Beta. For these labs we will be using the Classic UI, which you can access either by opening to the link given above or by selecting **Go to Classic UI**.

![Google_choose_Account](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/29568a4fe7828984db81564a3b7b1dcfe36ea098dc5f6611ec3d7bf098cd771f.png)

The BigQuery console appears:

![Google_choose_Account](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/5f2f0c00f29b3d1e93ffa0678f767f0cb497cad4a22c4ce09ce1a3091c159632.png)

1. Ensure that BigQuery is set to your **qwiklabs-gcp-\**\**\**\**** project. If it is, then proceed to the next step. If it’s not set to your project (for example, it might be set to another project like **Qwiklabs Resources**), click the drop down arrow next to the project name and then click **Switch to project**, then select your project as shown below:

![Google_choose_Account](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/040be43b7115e8425fbabe44c47aaf176f1df2bb11f4b51dafb55794f89997be.png)

Look at the left-side menu and you should see the **babyweight** dataset. Click on the blue down arrow to its left, and you should see your **prediction** table.

**Note:** If you do not see the prediction table, give it a few minutes as the pipeline has allowed-latency and that can add some delay.

![1fbaf89946687844.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/0fb15ee3b7340ff47b4472ba3b5941af144fe8a0f641e31c2a288620f4a42299.png)

### **Step 9**

Click on **Compose Query** button on the top left. Type the query below in the query box to retrieve rows from your predictions table. Click on **Show Options** button under the query box and uncheck **Use Legacy SQL** and click **Hide Options**.

```sql
#legacySQL
SELECT * FROM babyweight.predictions LIMIT 1000
```

![ccb8ccce73d92d9a.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/cac2d7bd7936df8038a80121a43260ee9ed6e9a661bcc670949b7f5abc9cdc66.png)

### **Step 10**

Click the **Run Query** button. Notice the **predicted_weights_pounds**column in the result.

![549d498cd2f18780.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/da6cbdf0f054a07167d5ca480faa10e972a60b272d249abafb6c954f5e1fe3a0.png)

### **Step 11**

Remember that your pipeline is still running. You can publish additional messages from your example.csv.gz and verify new rows added to your predictions table. Once you are satisfied, you may stop the Dataflow pipeline by going into your Dataflow Jobs page, and click the **Stop job**button on the right side Job summary window.

![69cff18d8f1cabb5.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/a7523aa059d414f07055241598fe3e9b66397d0fdecb7f8b0b48156ba45c3900.png)

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.



# Lab 3 (optional): [ML on GCP C7] Kubeflow End to End

## Introduction

![14df38356117980d.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/9a2385c13c078b62ee1bf0d9ef7dd0022a4afa9a2f096f7db7e8c672d49bfb0d.png)

[Kubeflow](https://www.kubeflow.org/) is a machine learning toolkit for [Kubernetes](https://kubernetes.io/). The project is dedicated to making **deployments** of machine learning (ML) workflows on Kubernetes simple, portable, and scalable. The goal is to provide a straightforward way to deploy best-of-breed open-source systems for ML to diverse infrastructures.

A Kubeflow deployment is:

* **Portable** - Works on any Kubernetes cluster, whether it lives on Google Cloud Platform (GCP), on-premise, or across providers.
* **Scalable** - Can utilize fluctuating resources and is only constrained by the number of resources allocated to the Kubernetes cluster.
* **Composable** - Enhanced with service workers to work offline or on low-quality networks

Kubeflow will let you organize loosely-coupled microservices as a single unit and deploy them to a variety of locations, whether that's a laptop or the cloud. This codelab will walk you through creating your own Kubeflow deployment.

### What you'll build

In this lab you're going to build a web app that summarizes GitHub issues using a trained model. Upon completion, your infrastructure will contain:

* A GKE cluster with standard Kubeflow and Seldon Core installations
* A training job that uses Tensorflow to generate a Keras model
* A serving container that provides predictions
* A UI that uses the trained model to provide summarizations for GitHub issues

### What you'll learn

* How to install [Kubeflow](https://github.com/kubeflow/kubeflow)
* How to run training using the [Tensorflow](https://www.tensorflow.org/) job server to generate a[Keras](https://keras.io/) model
* How to serve a trained model with [Seldon Core](https://github.com/SeldonIO/seldon-core)
* How to generate and use predictions from a trained model

### What you'll need

* A basic understanding of [Kubernetes](https://kubernetes.io/)
* A [GitHub](https://github.com/) account

## Setup the environment

### Qwiklabs setup

#### What you'll need

To complete this lab, you’ll need:

* Access to a standard internet browser (Chrome browser recommended).
* Time. Note the lab’s **Completion** time in Qwiklabs. This is an estimate of the time it should take to complete all steps. Plan your schedule so you have time to complete the lab. Once you start the lab, you will not be able to pause and return later (you begin at step 1 every time you start a lab).
* The lab's **Access** time is how long your lab resources will be available. If you finish your lab with access time still available, you will be able to explore the Google Cloud Platform or work on any section of the lab that was marked "if you have time". Once the Access time runs out, your lab will end and all resources will terminate.
* You **DO NOT** need a Google Cloud Platform account or project. An account, project and associated resources are provided to you as part of this lab.
* If you already have your own GCP account, make sure you do not use it for this lab.
* If your lab prompts you to log into the console, **use only the student account provided to you by the lab**. This prevents you from incurring charges for lab activities in your personal GCP account.

#### Start your lab

When you are ready, click **Start Lab**. You can track your lab’s progress with the status bar at the top of your screen.

**Important** What is happening during this time? Your lab is spinning up GCP resources for you behind the scenes, including an account, a project, resources within the project, and permission for you to control the resources needed to run the lab. This means that instead of spending time manually setting up a project and building resources from scratch as part of your lab, you can begin learning more quickly.

#### Find Your Lab’s GCP Username and Password

To access the resources and console for this lab, locate the Connection Details panel in Qwiklabs. Here you will find the account ID and password for the account you will use to log in to the Google Cloud Platform:

![Open Google Console](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/5eaee037e6cedbf49f6a702ab8a9ef820bb8e717332946ff76a0831a6396aafc.png)

If your lab provides other resource identifiers or connection-related information, it will appear on this panel as well.

#### Log in to Google Cloud Console

Using the Qwiklabs browser tab/window or the separate browser you are using for the Qwiklabs session, copy the Username from the Connection Details panel and click the **Open Google Console** button.

You'll be asked to Choose an account. Click **Use another account**. ![Google_choose_Account](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/c0da2885d2fec3205abd6021c643797ae3c7dae9498623b3a019668b5fa6192c.png)

Paste in the Username, and then the Password as prompted:

![Sign in to continue to Google Cloud Platform](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/acff356c737df55afb91a294edfab2d88b928f5ae38cd5e40f58ae9d1481581c.png)

Accept the terms and conditions.

Since this is a temporary account, which you will only have to access for this one lab:

* Do not add recovery options
* Do not sign up for free trials

**Note:** You can view the list of services by clicking the GCP Navigation menu button at the top-left next to “Google Cloud Platform”.![Cloud Console Menu](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/448b955585241ad55a5a1c6d20526e831cb2fce456418c3b3ab2d1d1b26545e2.png)

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

```bash
[core]
project = qwiklabs-gcp-9026c1edce19a407
Your active configuration is: [cloudshell-2366]

```

Close the main Navigation Menu by clicking the three lines at the top left of the screen (hamburger), next to the Google Cloud Platform logo.

### Enable Boost Mode

In the Cloud Shell window, click on the **Setting** icon at the far right. Select **Enable Boost Mode**, then **Restart Cloud Shell in Boost Mode**. This will provision a larger instance for your Cloud Shell session, resulting in speedier Docker builds.

![6b7d40bb615b632d.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/070154756bfc8e68d280a55e6d91394a512dfc2499ea314f84070c3f5c600768.png)

### Download the project files

The following commands in Cloud Shell to download and unpack an archive of the [Kubeflow examples repo](https://github.com/kubeflow/examples), which contains all of the official Kubeflow examples:

```bash
wget https://github.com/kubeflow/examples/archive/v0.2.zip
unzip v0.2.zip
mv examples-0.2 ${HOME}/examples
```

### Set your GitHub token

This lab involves the use of many different files obtained from public repos on GitHub. To prevent rate-limiting, setup an access token with no permissions. This is simply to authorize you as an individual rather than anonymous user.

1. Navigate to <https://github.com/settings/tokens> and generate a new token with no permissions.
2. Save it somewhere safe. If you lose it, you will need to delete and create a new one.
3. Set the GITHUB_TOKEN environment variable:

```bash
export \
  GITHUB_TOKEN=<token>
```

## Install Ksonnet

Set the correct version and an environment variable:

```bash
export KS_VER=ks_0.11.0_linux_amd64
```

## Install the binary

Download and unpack the appropriate binary, then add it to your $PATH:

```bash
wget -O /tmp/$KS_VER.tar.gz https://github.com/ksonnet/ksonnet/releases/download/v0.11.0/$KS_VER.tar.gz
mkdir -p ${HOME}/bin
tar -xvf /tmp/$KS_VER.tar.gz -C ${HOME}/bin
export PATH=$PATH:${HOME}/bin/$KS_VER
```

To familiarize yourself with Ksonnet concepts, see [this diagram](https://github.com/ksonnet/ksonnet/blob/master/docs/concepts.md#environment).

> The following diagram shows how the ksonnet framework works holistically:
> 
> [![ksonnet overview diagram](https://github.com/ksonnet/ksonnet/raw/master/docs/img/ksonnet_overview.svg?sanitize=true)](https://github.com/ksonnet/ksonnet/blob/master/docs/img/ksonnet_overview.svg)
> 
> ksonnet's package management schema can be summed up as follows:
> 
> *registry* > *package* > *prototype*
>  **Ex:** [`incubator` repo](https://github.com/ksonnet/parts/tree/master/incubator) > [Redis package](https://github.com/ksonnet/parts/tree/master/incubator/redis) > [`redis-stateless` prototype](https://github.com/ksonnet/parts/blob/master/incubator/redis/prototypes/redis-stateless.jsonnet)
> 
> **Application**
> 
> A ksonnet application represents a well-structured directory of Kubernetes [manifests](https://github.com/ksonnet/ksonnet/blob/master/docs/concepts.md#manifest). This directory is autogenerated by [`ks init`](https://github.com/ksonnet/ksonnet/blob/master/docs/cli-reference/ks_init.md).  These manifests typically tie together in some way—for example, they  might collectively define a web service like the following:
> 
>  [![ksonnet application diagram](https://github.com/ksonnet/ksonnet/raw/master/docs/img/guestbook_app.svg?sanitize=true)](https://github.com/ksonnet/ksonnet/blob/master/docs/img/guestbook_app.svg) 



### Retrieve the project ID

Store the project ID and activate the latest scopes:

```bash
export PROJECT_ID=$(gcloud config get-value project)
gcloud config set container/new_scopes_behavior true
```

## Create a service account

Create a service account with read/write access to storage buckets:

```bash
export SERVICE_ACCOUNT=github-issue-summarization
export SERVICE_ACCOUNT_EMAIL=${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com
gcloud iam service-accounts create ${SERVICE_ACCOUNT} \
  --display-name "GCP Service Account for use with kubeflow examples"

gcloud projects add-iam-policy-binding ${PROJECT_ID} --member \
  serviceAccount:${SERVICE_ACCOUNT_EMAIL} \
  --role=roles/storage.admin
```

Generate a credentials file for upload to the cluster:

```bash
export KEY_FILE=${HOME}/secrets/${SERVICE_ACCOUNT_EMAIL}.json
gcloud iam service-accounts keys create ${KEY_FILE} \
  --iam-account ${SERVICE_ACCOUNT_EMAIL}
```

## Create a storage bucket

Create a Cloud Storage bucket for storing your trained model and issue the “mb” (make bucket) command:

```bash
export BUCKET=kubeflow-${PROJECT_ID}
gsutil mb -c regional -l us-central1 gs://${BUCKET}
```

## Create a cluster

Create a managed Kubernetes cluster on Kubernetes Engine by running:

```bash
gcloud container clusters create kubeflow-qwiklab \
  --machine-type n1-standard-4 \
  --zone us-central1-a  \
  --scopes=compute-rw,storage-rw \
  --enable-autorepair
```

```
kubeconfig entry generated for kubeflow-qwiklab.
NAME              LOCATION       MASTER_VERSION  MASTER_IP      MACHINE_TYPE   NODE_VERSION  NUM_NODES  STATUS
kubeflow-qwiklab  us-central1-a  1.11.6-gke.2    35.226.236.74  n1-standard-4  1.11.6-gke.2  3          RUNNING
```



Cluster creation will take a few minutes to complete.

Connect your local environment to the Google Kubernetes Engine (GKE) cluster:

```bash
gcloud container clusters get-credentials kubeflow-qwiklab --zone us-central1-a
```

This configures your `kubectl` context so that you can interact with your cluster. To verify the connection, run the following command:

```bash
kubectl cluster-info
```

```
Kubernetes master is running at https://35.226.236.74
GLBCDefaultBackend is running at https://35.226.236.74/api/v1/namespaces/kube-system/services/default-http-backend:http/proxy
Heapster is running at https://35.226.236.74/api/v1/namespaces/kube-system/services/heapster/proxy
KubeDNS is running at https://35.226.236.74/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy
Metrics-server is running at https://35.226.236.74/api/v1/namespaces/kube-system/services/https:metrics-server:/proxy
To further debug and diagnose cluster problems, use 'kubectl cluster-info dump'.

```



Verify that this IP address matches the IP address corresponding to the Endpoint in your [Google Cloud Platform Console](https://console.cloud.google.com/kubernetes/clusters/details/us-central1-a/kubeflow-qwiklab) or by comparing the Kubernetes master IP is the same as the Master_IP address in the previous step.

To enable the installation of Kubeflow and Seldon components, run the following to create two ClusterRoleBindings, which allows the creation of objects:

```bash
kubectl create clusterrolebinding default-admin \
  --clusterrole=cluster-admin \
  --user=$(gcloud config get-value account)
kubectl create clusterrolebinding seldon-admin \
  --clusterrole=cluster-admin \
  --serviceaccount=default:default
```

Upload service account credentials:

```bash
kubectl create secret generic user-gcp-sa \
  --from-file=user-gcp-sa.json="${KEY_FILE}"
```

## Install Kubeflow with Seldon

Ksonnet is a templating framework, which allows us to utilize common object definitions and customize them to our environment. We begin by referencing Kubeflow templates and apply environment-specific parameters. Once manifests have been generated specifically for our cluster, they can be applied like any other kubernetes object using `kubectl`.

### Initialize a ksonnet app

Run these commands to go inside the `github_issue_summarization`directory; then create an new ksonnet app directory, fill it with boilerplate code, and retrieve component files:

```bash
cd ${HOME}/examples/github_issue_summarization
ks init kubeflow
cd kubeflow
cp ../ks-kubeflow/components/kubeflow-core.jsonnet components
cp ../ks-kubeflow/components/params.libsonnet components
cp ../ks-kubeflow/components/seldon.jsonnet components
cp ../ks-kubeflow/components/tfjob-v1alpha2.* components
cp ../ks-kubeflow/components/ui.* components
```

### Install packages and generate core components

Register the Kubeflow template repository:

```bash
export VERSION=v0.2.0-rc.1
ks registry add kubeflow github.com/kubeflow/kubeflow/tree/${VERSION}/kubeflow
```

Install Kubeflow core and Seldon components:

```bash
ks pkg install kubeflow/core@${VERSION}
ks pkg install kubeflow/tf-serving@${VERSION}
ks pkg install kubeflow/tf-job@${VERSION}
ks pkg install kubeflow/seldon@${VERSION}
```

**Note:** If you run into rate-limit errors, be sure your GITHUB_TOKEN environment variable is set properly. See the **Set Your GitHub Token**section above for more details.

## Create the environment

Define an environment that references our specific cluster:

```bash
ks env add gke
ks param set --env gke kubeflow-core \
  cloud "gke"
ks param set --env gke kubeflow-core \
  tfAmbassadorServiceType "LoadBalancer"
```

Apply the generated manifests to the cluster to create the Kubeflow and Seldon components:

```bash
ks apply gke -c kubeflow-core -c seldon
```

Your cluster now contains a Kubeflow installation with Seldon with the following components:

* Reverse HTTP proxy (Ambassador)
* Central dashboard
* Jupyterhub
* TF job dashboard
* TF job operator
* Seldon cluster manager
* Seldon cache

You can view the components by running:

```bash
kubectl get pods
```

**Note:** Image pulls can take a while. You can expect pods to remain in ContainerCreating status for a few minutes.

You should see output similar to this:

![b955db44cfb34944.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/04e04c5f7ddfd8d3c445ddaf432c7013ea7ef61a59f0ddfe5e48f4eeab6801b8.jpg)

```
NAME                                        READY     STATUS              RESTARTS   AGE
ambassador-56ccfbc448-5kr9m                 0/2       ContainerCreating   0          31s
ambassador-56ccfbc448-7mcf7                 0/2       ContainerCreating   0          31s
ambassador-56ccfbc448-qptq8                 0/2       ContainerCreating   0          31s
centraldashboard-6d5c645b69-79725           1/1       Running             0          29s
redis-5b44cd7974-5fdq6                      1/1       Running             0          42s
seldon-cluster-manager-6cdd57d68d-mcmpk     1/1       Running             0          42s
tf-hub-0                                    0/1       ContainerCreating   0          30s
tf-job-dashboard-7db5b474c8-w5wvm           0/1       ContainerCreating   0          30s
tf-job-operator-v1alpha2-7954bdcf76-djmcs   0/1       ContainerCreating   0          30s
```



## Train a model

In this section, you will create a component that trains a model.

Set the component parameters:

```bash
cd ${HOME}/examples/github_issue_summarization/kubeflow
ks param set --env gke tfjob-v1alpha2 image "gcr.io/kubeflow-examples/tf-job-issue-summarization:v20180629-v0.1-2-g98ed4b4-dirty-182929"
ks param set --env gke tfjob-v1alpha2 output_model_gcs_bucket "${BUCKET}"
```

The training component `tfjob-v1alpha2` is now configured to use a pre-built image. If you would prefer to generate your own instead, continue with the Optional create the training image step.

## (Optional) Create the training image

Image creation can take 5-10 minutes.

In the `github_issue_summarization` directory, navigate to the folder containing the training code (`notebooks`). From there, issue a `make` command that builds the image and stores it in Google Container Registry (GCR). This places it in a location accessible from inside the cluster.

```
cd ${HOME}/examples/github_issue_summarization/notebooks
make PROJECT=${PROJECT_ID} push
```

Once the image has been built and stored in GCR, update the component parameter with a link that points to the custom image:

```
export TAG=$(gcloud container images list-tags \
gcr.io/${PROJECT_ID}/tf-job-issue-summarization \
--limit=1 \
--format='get(tags)')
ks param set --env gke tfjob-v1alpha2 image "gcr.io/${PROJECT_ID}/tf-job-issue-summarization:${TAG}"
```

### Launch training

Apply the component manifests to the cluster:

```bash
ks apply gke -c tfjob-v1alpha2
```

### View the running job

View the resulting pods:

```bash
kubectl get pods
```

Your cluster state should look similar to this:

![8ef9030596923b35.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/119f8adf9f43c00e3d504cea3aba521ee2ecb00f78d7b2c7b5a4d22dcab16d72.jpg)

```
NAME                                        READY     STATUS              RESTARTS   AGE
ambassador-56ccfbc448-5kr9m                 2/2       Running             0          3m
ambassador-56ccfbc448-7mcf7                 2/2       Running             0          3m
ambassador-56ccfbc448-qptq8                 2/2       Running             0          3m
centraldashboard-6d5c645b69-79725           1/1       Running             0          3m
redis-5b44cd7974-5fdq6                      1/1       Running             0          3m
seldon-cluster-manager-6cdd57d68d-mcmpk     1/1       Running             0          3m
tf-hub-0                                    1/1       Running             0          3m
tf-job-dashboard-7db5b474c8-w5wvm           1/1       Running             0          3m
tf-job-operator-v1alpha2-7954bdcf76-djmcs   1/1       Running             0          3m
tfjob-issue-summarization-master-0          0/1       ContainerCreating   0          33s
```

...

```
tfjob-issue-summarization-master-0          1/1       Running   0          2m
```



It can take a few minutes to pull the image and start the container.

Once the new pod is running, tail the logs:

```bash
kubectl logs -f \
  $(kubectl get pods -ltf_job_key=tfjob-issue-summarization -o=jsonpath='{.items[0].metadata.name}')
```

Inside the pod, you will see the download of source data (`github-issues.zip`) before training begins. Continue tailing the logs until the pod exits on its own and you find yourself back at the command prompt. When you see the command prompt, continue with the next step.

![62c11bb1d8773d26.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/5d0b5c23e5908142acc8faa17d8a843444e078ea094cc463ba30e9b9b585e915.png)

```
Using TensorFlow backend.
INFO|2019-02-19T10:14:25|/workdir/train.py|108| Namespace(input_data='', input_data_gcs_bucket='kubeflow-examples', input_data_gcs_path='github-issue-summarization-data/github-issues.zip', learning_rate='0.001', output_body_preprocessor_dpkl='body_pp.dpkl', output_model='', output_model_gcs_bucket='kubeflow-qwiklabs-gcp-9026c1edce19a407', output_model_gcs_path='github-issue-summarization-data', output_model_h5='seq2seq_model_tutorial.h5', output_title_preprocessor_dpkl='title_pp.dpkl', output_train_body_vecs_npy='train_body_vecs.npy', output_train_title_vecs_npy='train_title_vecs.npy', sample_size=100000)
INFO|2019-02-19T10:14:25|/workdir/train.py|129| Download bucket kubeflow-examples object github-issue-summarization-data/github-issues.zip.
INFO|2019-02-19T10:15:54|/workdir/train.py|151| Train: 90000 rows 3 columns
INFO|2019-02-19T10:15:54|/workdir/train.py|152| Test: 10000 rows 3 columns
WARNING|2019-02-19T10:15:54|/opt/conda/lib/python3.6/site-packages/ktext/preprocess.py|279| ....tokenizing data
WARNING|2019-02-19T10:17:02|/opt/conda/lib/python3.6/site-packages/ktext/preprocess.py|289| (1/2) done. 68 sec
WARNING|2019-02-19T10:17:02|/opt/conda/lib/python3.6/site-packages/ktext/preprocess.py|290| ....building corpus
WARNING|2019-02-19T10:17:08|/opt/conda/lib/python3.6/site-packages/ktext/preprocess.py|301| (2/2) done. 5 sec
WARNING|2019-02-19T10:17:08|/opt/conda/lib/python3.6/site-packages/ktext/preprocess.py|302| Finished parsing 90,000 documents.
WARNING|2019-02-19T10:17:08|/opt/conda/lib/python3.6/site-packages/ktext/preprocess.py|340| ...fit is finished, beginning transform
WARNING|2019-02-19T10:17:11|/opt/conda/lib/python3.6/site-packages/ktext/preprocess.py|343| ...padding data
WARNING|2019-02-19T10:17:12|/opt/conda/lib/python3.6/site-packages/ktext/preprocess.py|345| done. 3 sec
INFO|2019-02-19T10:17:12|/workdir/train.py|163| Example original body: challenge escape sequences in strings https://www.freecodecamp.com/challenges/escape-sequences-in-strings ?solution=%0avar%20mystr%20%3d%20%22firstline%5cn%5c%5cseondline%5c%5c%5crthirdline%22%3b%0a%0a%0a has an issue. user agent is: <code>mozilla/5.0 windows nt 10.0; wow64 applewebkit/537.36 khtml, like gecko chrome/55.0.2883.87 safari/537.36</code>. please describe how to reproduce this issue, and include links to screenshots if possible. my code: javascript var mystr = firstline \\seondline\\\rthirdline ;
INFO|2019-02-19T10:17:12|/workdir/train.py|164| Example body after pre-processing: [2376 2837 4118    7 1440   12  457    1    1    2 2192    2    1 6796
    1    1    1    1    1    2 4861 3346 3346 3346   73   36   42   61
  897    8   52 1222    2  177 1227    2 3687 2882    2 2883   49 1569
  336    2 7544 4651 1075    2   52   79  805   70    4  144   13   42
    9  247  621    4 1291   21  164   39   52  595  262    1    1    1]
WARNING|2019-02-19T10:17:12|/opt/conda/lib/python3.6/site-packages/ktext/preprocess.py|279| ....tokenizing data
WARNING|2019-02-19T10:17:23|/opt/conda/lib/python3.6/site-packages/ktext/preprocess.py|289| (1/2) done. 11 sec
WARNING|2019-02-19T10:17:23|/opt/conda/lib/python3.6/site-packages/ktext/preprocess.py|290| ....building corpus
WARNING|2019-02-19T10:17:24|/opt/conda/lib/python3.6/site-packages/ktext/preprocess.py|301| (2/2) done. 1 sec
WARNING|2019-02-19T10:17:24|/opt/conda/lib/python3.6/site-packages/ktext/preprocess.py|302| Finished parsing 90,000 documents.
WARNING|2019-02-19T10:17:24|/opt/conda/lib/python3.6/site-packages/ktext/preprocess.py|340| ...fit is finished, beginning transform
WARNING|2019-02-19T10:17:25|/opt/conda/lib/python3.6/site-packages/ktext/preprocess.py|343| ...padding data
WARNING|2019-02-19T10:17:25|/opt/conda/lib/python3.6/site-packages/ktext/preprocess.py|345| done. 1 sec
INFO|2019-02-19T10:17:25|/workdir/train.py|173| Example original title: platform refusing to accept correct response to excersise
INFO|2019-02-19T10:17:25|/workdir/train.py|174| Example title after pre-processing: [  2 705   1   4 830 486 375   4   1   3   0   0]
2019-02-19 10:17:31.045010: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
/opt/conda/lib/python3.6/site-packages/keras/engine/network.py:888: UserWarning: Layer Decoder-GRU was passed non-serializable keyword arguments: {'initial_state': [<tf.Tensor 'Encoder-Model/Encoder-Last-GRU/while/Exit_3:0' shape=(?, 300) dtype=float32>]}. They will not be included in the serialized model (and thus will be missing at deserialization time).
  '. They will not be included '
INFO|2019-02-19T10:17:31|/workdir/train.py|278| Uploading model files to bucket kubeflow-qwiklabs-gcp-9026c1edce19a407 path github-issue-summarization-data.
Shape of encoder input: (90000, 70)
Size of vocabulary for body_pp.dpkl: 8002
Size of vocabulary for title_pp.dpkl: 4502
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
Decoder-Input (InputLayer)      (None, None)         0
__________________________________________________________________________________________________
Decoder-Word-Embedding (Embeddi (None, None, 300)    1350600     Decoder-Input[0][0]
__________________________________________________________________________________________________
Encoder-Input (InputLayer)      (None, 70)           0
__________________________________________________________________________________________________
Decoder-Batchnorm-1 (BatchNorma (None, None, 300)    1200        Decoder-Word-Embedding[0][0]
__________________________________________________________________________________________________
Encoder-Model (Model)           (None, 300)          2942700     Encoder-Input[0][0]
__________________________________________________________________________________________________
Decoder-GRU (GRU)               [(None, None, 300),  540900      Decoder-Batchnorm-1[0][0]
                                                                 Encoder-Model[1][0]
__________________________________________________________________________________________________
Decoder-Batchnorm-2 (BatchNorma (None, None, 300)    1200        Decoder-GRU[0][0]
__________________________________________________________________________________________________
Final-Output-Dense (Dense)      (None, None, 4502)   1355102     Decoder-Batchnorm-2[0][0]
==================================================================================================
Total params: 6,191,702
Trainable params: 6,189,902
Non-trainable params: 1,800
__________________________________________________________________________________________________
```



To verify that training completed successfully, check to make sure all three model files were uploaded to your GCS bucket:

```bash
gsutil ls gs://${BUCKET}/github-issue-summarization-data
```

You should see something like this:

![8b3c95b353a11b6b.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/041d20f44898e5f55538ca8a5b4f32f9e4a1964bc7596c1a8036d3c79d17675d.png)

```
gs://kubeflow-qwiklabs-gcp-9026c1edce19a407/github-issue-summarization-data/body_pp.dpkl
gs://kubeflow-qwiklabs-gcp-9026c1edce19a407/github-issue-summarization-data/seq2seq_model_tutorial.h5
gs://kubeflow-qwiklabs-gcp-9026c1edce19a407/github-issue-summarization-data/title_pp.dpkl
```



## Serve the trained model

In this section, you will create a component that serves a trained model.

Set component parameters:

```bash
export SERVING_IMAGE=gcr.io/kubeflow-examples/issue-summarization-model:v20180629-v0.1-2-g98ed4b4-dirty-182929
```

### Create the serving image

The serving component is configured to run a pre-built image, to save you some time. If you would prefer to serve the model you created in the previous step, you can generate your own by continuing with *Optional image creation* step. Otherwise, continue with the *Create the serving component* step.

### **(Optional) image creation**

#### **Download the trained model files**

Retrieve the trained model files that were generated in the previous step:

```
cd ${HOME}/examples/github_issue_summarization/notebooks``gsutil cp gs://${BUCKET}/github-issue-summarization-data/* .
```

#### **Generate image build files**

Using a Seldon wrapper, generate files for building a serving image. This command creates a build directory and image creation script:

```
docker run -v $(pwd):/my_model seldonio/core-python-wrapper:0.7 \
/my_model IssueSummarization 0.1 gcr.io \
--base-image=python:3.6 \
--image-name=${PROJECT_ID}/issue-summarization-model
```

#### **Generate a serving image**

Using the files created by the wrapper, generate a serving image and store it in GCR:

```
cd ${HOME}/examples/github_issue_summarization/notebooks/build
./build_image.sh
gcloud docker -- push gcr.io/${PROJECT_ID}/issue-summarization-model:0.1
export SERVING_IMAGE=gcr.io/${PROJECT_ID}/issue-summarization-model:0.1
```

### Create the serving component

This serving component is configured to run a pre-built image. Using a Seldon ksonnet template, generate the serving component.

Navigate back to the ksonnet app directory and issue the following command:

```bash
cd ${HOME}/examples/github_issue_summarization/kubeflow
ks generate seldon-serve-simple issue-summarization-model \
  --name=issue-summarization \
  --image=${SERVING_IMAGE} \
  --replicas=2
```

### Launch serving

Apply the component manifests to the cluster:

```bash
ks apply gke -c issue-summarization-model
```

### View the running pods

You will see several new pods appear:

```bash
kubectl get pods
```

Your cluster state should look similar to this:

![ad0c22eb9fa07f31.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/0341bfde6dd7169f9b53c2d62173e2459b80338759cb0a3dbbdd27b6717a3cfa.png)

```
NAME                                                       READY     STATUS              RESTARTS   AGE
ambassador-56ccfbc448-5kr9m                                2/2       Running             0          14m
ambassador-56ccfbc448-7mcf7                                2/2       Running             0          14m
ambassador-56ccfbc448-qptq8                                2/2       Running             0          14m
centraldashboard-6d5c645b69-79725                          1/1       Running             0          14m
issue-summarization-issue-summarization-676dc95579-qnc7w   0/2       ContainerCreating   0          21s
issue-summarization-issue-summarization-676dc95579-tl2dn   0/2       ContainerCreating   0          21s
redis-5b44cd7974-5fdq6                                     1/1       Running             0          15m
seldon-cluster-manager-6cdd57d68d-mcmpk                    1/1       Running             0          15m
tf-hub-0                                                   1/1       Running             0          14m
tf-job-dashboard-7db5b474c8-w5wvm                          1/1       Running             0          14m
tf-job-operator-v1alpha2-7954bdcf76-djmcs                  1/1       Running             0          14m
tfjob-issue-summarization-master-0                         0/1       Completed           0          11m
```



Wait a minute or two and re-run the previous command. Once the pods are running, tail the logs for one of the serving containers to verify that it is running on port 9000:

```bash
kubectl logs \
  $(kubectl get pods \
    -lseldon-app=issue-summarization \
    -o=jsonpath='{.items[0].metadata.name}') \
  issue-summarization
```

```
Using TensorFlow backend.
body_pp file body_pp.dpkl
title_pp file title_pp.dpkl
model file seq2seq_model_tutorial.h5
2019-02-19 10:24:45.078172: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
 * Running on http://0.0.0.0:9000/ (Press CTRL+C to quit)
```

Press **Ctrl** + **C** to return to the command line.

## Add a UI

### Set parameter values

```bash
cd ${HOME}/examples/github_issue_summarization/kubeflow
ks param set --env gke ui image "gcr.io/kubeflow-examples/issue-summarization-ui:v20180629-v0.1-2-g98ed4b4-dirty-182929"
ks param set --env gke ui githubToken ${GITHUB_TOKEN}
ks param set --env gke ui modelUrl "http://issue-summarization.default.svc.cluster.local:8000/api/v0.1/predictions"
ks param set --env gke ui serviceType "LoadBalancer"
```

### (Optional) Create the UI image

The UI component is now configured to use a pre-built image. If you would prefer to generate your own instead, continue with this step.

**Note:** Image creation can take 5-10 minutes. This step is optional. Alternatively, skip directly to the **Launch the UI** section below.

Switch to the docker directory and build the image for the UI:

```bash
cd ${HOME}/examples/github_issue_summarization/docker
docker build -t gcr.io/${PROJECT_ID}/issue-summarization-ui:latest .
```

After it has been successfully built, store it in GCR:

```bash
gcloud docker -- push gcr.io/${PROJECT_ID}/issue-summarization-ui:latest
```

Update the component parameter with a link that points to the custom image:

```bash
cd ${HOME}/examples/github_issue_summarization/kubeflow
ks param set --env gke ui image gcr.io/${PROJECT_ID}/issue-summarization-ui:latest
```

### Launch the UI

Apply the component manifests to the cluster:

```bash
ks apply gke -c ui
```

You should see an additional pod, it's status will be ContainerCreating:

![36ee1c18b4feb688.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/fe48bd9354add1c670e9c85b103b13917b5206100a631347b280a1f15447653b.png)

```
NAME                                                       READY     STATUS      RESTARTS   AGE
ambassador-56ccfbc448-5kr9m                                2/2       Running     0          19m
ambassador-56ccfbc448-7mcf7                                2/2       Running     0          19m
ambassador-56ccfbc448-qptq8                                2/2       Running     0          19m
centraldashboard-6d5c645b69-79725                          1/1       Running     0          19m
issue-summarization-issue-summarization-676dc95579-qnc7w   2/2       Running     0          4m
issue-summarization-issue-summarization-676dc95579-tl2dn   2/2       Running     0          4m
issue-summarization-ui-7c5b7b7b9c-dgt86                    1/1       Running     0          1m
redis-5b44cd7974-5fdq6                                     1/1       Running     0          19m
seldon-cluster-manager-6cdd57d68d-mcmpk                    1/1       Running     0          19m
tf-hub-0                                                   1/1       Running     0          19m
tf-job-dashboard-7db5b474c8-w5wvm                          1/1       Running     0          19m
tf-job-operator-v1alpha2-7954bdcf76-djmcs                  1/1       Running     0          19m
tfjob-issue-summarization-master-0                         0/1       Completed   0          16m
```

### View the UI

To view the UI, get the external IP address:

```bash
kubectl get svc issue-summarization-ui
```

Wait until the external IP address has been populated. Re-run the command until it appears. Copy the External-IP address.

![837bb38a0b5bd625.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/2614c0c12d49ea4cbef64d87524ffc24413f5885124f4f582d91feec2a105d3c.png)

```
NAME                     TYPE           CLUSTER-IP      EXTERNAL-IP      PORT(S)        AGE
issue-summarization-ui   LoadBalancer   10.43.252.251   35.192.156.198   80:30356/TCP   1m
```

In a browser, paste the `EXTERNAL-IP` to view the results. You should see something like this:

![f783cfbd99ecbc94.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/0631907e78af51ec8d8b63a2053718aa61b151c43c2312e01dd56f6042efd308.png)

Click the **Populate Random Issue** button to fill in the large text box with a random issue summary. Then click the **Generate Title** button to view the machine generated title produced by your trained model. Click the button a couple of times to give yourself some more data to look at in the next step.

<https://github.com/hamelsmu/Seq2Seq_Tutorial/blob/master/notebooks/Tutorial.ipynb>

## View serving container logs

In Cloud Shell, tail the logs of one of the serving containers to verify that it is receiving a request from the UI and providing a prediction in response:

```bash
kubectl logs -f \
  $(kubectl get pods \
    -lseldon-app=issue-summarization \
    -o=jsonpath='{.items[0].metadata.name}') \
  issue-summarization
```

Back in the UI, press the **Generate Title** button a few times to view the POST request in Cloud Shell. Since there are two serving containers, you might need to try a few times before you see the log entry.

Press Ctrl+C to return to the command prompt.

![80e4cb7771ebb9f5.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/46c63457781e040471fd88bdb092987376950bdcdaf4af17242e4ab2d04fc732.png)

```
Using TensorFlow backend.
body_pp file body_pp.dpkl
title_pp file title_pp.dpkl
model file seq2seq_model_tutorial.h5
2019-02-19 10:24:45.078172: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
 * Running on http://0.0.0.0:9000/ (Press CTRL+C to quit)
10.40.0.8 - - [19/Feb/2019 10:30:49] "POST /predict HTTP/1.1" 200 -
10.40.0.8 - - [19/Feb/2019 10:31:09] "POST /predict HTTP/1.1" 200 -
10.40.0.8 - - [19/Feb/2019 10:31:40] "POST /predict HTTP/1.1" 200 -
```

## Clean up

### Remove GitHub token

Navigate to <https://github.com/settings/tokens> and remove the generated token.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.



# Additional Resources

# GOOGLE CLOUD PLATFORM: Simplifying machine learning on open hybrid clouds with Kubeflow

<https://cloud.google.com/blog/products/gcp/simplifying-machine-learning-on-open-hybrid-clouds-with-kubeflow>



Kirat Pandya

Hybrid ML Technical Lead, Google Cloud

March 29, 2018

Traditional machine learning (ML) deployments are complicated, and moving a model and its associated dependencies from a developer’s laptop to a cloud cluster is usually an involved process. Last year, we released [Kubeflow](http://blog.kubernetes.io/2017/12/introducing-kubeflow-composable.html), an open-source project to make it easier to use machine learning software stacks like TensorFlow, Scikit-Learn, and others, all on Kubernetes. The Kubeflow framework ties infrastructure and machine learning solutions together, and since the launch, has earned almost 3,000 stars on [its GitHub repository](https://github.com/kubeflow/kubeflow).

With a growing customer demand for ML solutions, we’ve been working closely with partners to bring comprehensively tested and supported offerings for the open hybrid cloud—ones that can run on-premises or in the cloud.

Cisco and Google Cloud have been working on an [open hybrid cloud architecture](https://cloud.google.com/cisco/) to help customers maximize their investments across cloud and on-premises environments. Continuing this commitment, Cisco [announced](https://blogs.cisco.com/datacenter/cisco-ucs-and-hyperflex-for-ai-ml-workloads-in-the-data-center) that the Unified Computing System (UCS) and HyperFlex platforms will leverage Kubeflow to provide production-grade on-premise infrastructure to run AI/ML jobs.

![kubeflow-cisco-15rk6.JPEG](https://storage.googleapis.com/gweb-cloudblog-publish/images/kubeflow-cisco-15rk6.max-600x600.JPEG)![img](https://cloud.google.com/blog/)



![kubeflow-cisco-2mfd0.JPEG](https://storage.googleapis.com/gweb-cloudblog-publish/images/kubeflow-cisco-2mfd0.max-600x600.JPEG)![img](https://cloud.google.com/blog/)HyperFlex and Unified Computing System (UCS) platforms, for Cloud and on-prem, respectively

Kubeflow will be the underlying deployment infrastructure for these Cisco platforms, which handle non-trivial and tedious tasks such as driver versioning, Kubernetes bringup, and Kubeflow setup, so that customers can focus on machine learning rather than managing infrastructure.

This joint Cisco and Google solution brings Google’s leading machine learning capabilities and frameworks to the enterprise—providing businesses the ability to easily and quickly deploy AI or ML workloads in their own data center and reduce the time to insights.

—Kaustubh Das, Vice President of the Computing Systems Product Group at Cisco

We’re also working with the solution provider [H2O.ai](https://www.h2o.ai/) to bring [H2O-3](https://www.h2o.ai/h2o/) and [Driverless AI](https://www.h2o.ai/driverless-ai/) to Kubeflow, so it can run across open hybrid infrastructure. You can try H2O-3 on Kubeflow by following the instructions [here](https://blog.h2o.ai/2018/03/h2o-kubeflow-kubernetes-how-to/).

We’re excited to be one of the early contributors to Kubeflow, which makes it easy for machine learning workloads to be easily scaled from on-prem to cloud. We continue to democratize AI for enterprises and with Kubeflow, continue on our mission to build a strong open source AI community.

—Vinod Iyengar, Director of Partnerships and Alliances at H2O.ai



### Building your own open hybrid cloud

To get started with Kubeflow, try it out in your browser with Katacoda. Then, stay up-to-date on the latest updates via the Slack channel kubeflow-discuss email list (Google Group), and
Twitter. We believe that our holistic and open approach for Kubeflow can help solution and infrastructure partners deliver better offerings to customers. To learn more about how you can leverage Kubeflow to make your offerings ready for the open hybrid cloud, reach out to our

Technology and Services Partnerships team.

POSTED IN:[GOOGLE CLOUD PLATFORM](https://cloud.google.com/blog/products/gcp)[AI & MACHINE LEARNING](https://cloud.google.com/blog/products/ai-machine-learning)



# Introduction to Cloud ML Engine

<https://cloud.google.com/ml-engine/docs/tensorflow/technical-overview>

Use Cloud ML Engine to train your machine learning models at scale, to host your trained model in the cloud, and to use your model to make predictions about new data.

## A brief description of machine learning

Machine learning (ML) is a subfield of artificial intelligence (AI). The goal of ML is to make computers learn from the data that you give them. Instead of writing code that describes the action the computer should take, your code provides an algorithm that adapts based on examples of intended behavior. The resulting program, consisting of the algorithm and associated learned parameters, is called a trained model.

## Where Cloud ML Engine fits in the ML workflow

The diagram below gives a high-level overview of the stages in an ML workflow. The blue-filled boxes indicate where Cloud ML Engine provides managed services and APIs:

[![ML workflow](https://cloud.google.com/ml-engine/docs/images/ml-workflow.svg)](https://cloud.google.com/ml-engine/docs/images/ml-workflow.svg)ML workflow

As the diagram indicates, you can use Cloud ML Engine to manage the following stages in the ML workflow:

* Train an ML model on your data:
  * Train model
  * Evaluate model accuracy
  * Tune hyperparameters
* Deploy your trained model.
* Send prediction requests to your model:
  * Online prediction
  * Batch prediction
* Monitor the predictions on an ongoing basis.
* Manage your models and model versions.

## Components of Cloud ML Engine

This section describes the pieces that make up Cloud ML Engine and the primary purpose of each piece.

### Google Cloud Platform Console

You can deploy models to the cloud and manage your models, versions, and jobs on the [GCP Console](https://console.cloud.google.com/mlengine/models). This option gives you a user interface for working with your machine learning resources. As part of GCP, your Cloud ML Engine resources are connected to useful tools like Stackdriver Logging and Stackdriver Monitoring.

### The `gcloud` command-line tool

You can manage your models and versions, submit jobs, and accomplish other Cloud ML Engine tasks at the command line with the [`gcloud ml-engine` command-line tool](https://cloud.google.com/sdk/gcloud/reference/ml-engine/).

We recommend `gcloud` commands for most Cloud ML Engine tasks, and the REST API (see below) for online predictions.

### REST API

The Cloud ML Engine [REST API](https://cloud.google.com/ml-engine/reference/rest/) provides RESTful services for managing jobs, models, and versions, and for making predictions with hosted models on GCP.

You can use the [Google APIs Client Library for Python](https://developers.google.com/api-client-library/python/start/installation) to access the APIs. When using the client library, you use Python representations of the resources and objects used by the API. This is easier and requires less code than working directly with HTTP requests.

We recommend the REST API for serving online predictions in particular.

## What's next

* Walk through an end-to-end example of training a model, uploading the model to the cloud, and sending prediction requests, in the [getting-started guide](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction).
* View the full set of [guides](https://cloud.google.com/ml-engine/docs/tensorflow/).

 



# [[eof]]



[[todo]]:

* make python and html files of all notebooks in all courses!
* make pdfs of all .md-files in all courses! (add to .gitignore)