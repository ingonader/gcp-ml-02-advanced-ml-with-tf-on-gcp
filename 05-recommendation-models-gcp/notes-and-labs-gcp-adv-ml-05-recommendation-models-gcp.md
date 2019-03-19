# Recommendation Systems with Tensorflow on GCP



## to download materials

software to download materials: <https://github.com/coursera-dl/coursera-dl>

link to coursera material: <https://www.coursera.org/learn/end-to-end-ml-tensorflow-gcp/lecture/wWBkE/effective-ml>

```bash
coursera-dl -u "ingo.nader@gmail.com" recommendation-models-gcp -sl en --download-quizzes --download-notebooks --about

#coursera-dl -u "ingo.nader@gmail.com" recommendation-models-gcp -sl "en,de" --download-quizzes --download-notebooks --about
```

Done with modified version of coursera-dl: <https://github.com/coursera-dl/coursera-dl/issues/702>



## Notes

* Do labs in Google Chrome (recommended)
* Open Incognito window, go to <http://console.cloud.google.com/>



# Lab 1: [ML on GCP C10] Content-Based Filtering by Hand

1 hour 30 minutes

1 Credit





Rate Lab

## Overview

This lab shows you how to do content-based filtering using low-level TensorFlow commands.

## Objectives

In this lab, you learn to perform the following tasks:

* Create and compute a user feature matrix
* Compute where each user lies in the feature embedding space
* Create recommendations for new movies based on similarity measures between the user and movie feature vectors.

## Introduction

In this lab, you'll be providing movie recommendations for a set of users. Content-based filtering uses features of the items and users to generate recommendations. In this small example, we'll be using low-level TensorFlow operations and a very small set of movies and users to illustrate how this occurs in larger content based recommendation system.

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

## Launch Cloud Datalab

To launch Cloud Datalab:

1. In **Cloud Shell**, type:

```
gcloud compute zones list
```

1. Pick a zone in a geographically closeby region.
2. In **Cloud Shell**, type:

```
datalab create bdmlvm --zone <ZONE>
```

Datalab will take about 5 minutes to start.

```
Note: follow the prompts during this process.
```

## Checkout notebook into Cloud Datalab

If necessary, wait for Datalab to finish launching. Datalab is ready when you see a message prompting you to do a "Web Preview".

1. Click on the **Web Preview** icon on the top-right corner of the Cloud Shell ribbon. Click on the **Change port**. Switch to port **8081** using the **Change Preview Port** dialog box, and then click on **Change and Preview**.

   ![ChangePort.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/0fe2e17d4078e43c572498391788db31ddc98129a614c51db9fb90116ba4a142.png)

   ![ChangePreviewPort.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/ff07554423a417f49f859aaa5fe3a7ac6bcfc3d3db9add1ab99018c681d74938.png)

   Note: The connection to your Datalab instance remains open for as long as the datalab command is active. If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command **`datalab connect bdmlvm`** in your new Cloud Shell.

To clone the course repo in your datalab instance:

**Step 1**

In Cloud Datalab home page (browser), navigate into **notebooks** and add a new notebook using the icon ![notebook.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/fef0cc8c36a1856aa4ca73423f2ba59dde635267437c1253c268f366dfe19899.png) on the top left.

**Step 2**

Rename this notebook as **repocheckout**.

**Step 3**

In the new notebook, enter the following commands in the cell, and click on **Run** (on the top navigation bar) to run the commands:

```
%bash
git clone https://github.com/GoogleCloudPlatform/training-data-analyst
rm -rf training-data-analyst/.git
```

![clone.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/0a35d9ea37ae5908d89379a143c4fcd6292a6d29819fd34bc097ae17f21bd875.png)

**Step 4**

Confirm that you have cloned the repo by going back to Datalab browser, and ensure you see the **training-data-analyst** directory. All the files for all labs throughout this course are available in this directory.

![datalab.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/821c529680587fbc93135a3edde224a523aa07d0c38a07cf7967f13d082b7f0e.png)

## Open a Datalab notebook

1. In the Datalab browser, navigate to **datalab** > **notebooks** > **training-data-analyst** > **courses** > **machine_learning** > **deepdive** > **10_recommend** > **labs** and open **content_based_by_hand.ipynb**.
2. Read the commentary, **Click Clear** | **Clear all Cells**, then run the Python snippets (Use **Shift+Enter** to run each piece of code) in the cell, step by step.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.









# [[eof]]

[[todo]]:

* make python and html files of all **notebooks** in all courses!
* make pdfs of all **.md-files i**n all courses! (add to .gitignore)
* make "final" model with all bits and pieces:
  * tensorboard infos that work
  * get model from cloud storage to local disk
  * load model from disk and make predictions
  * get "local" evaluation metric

