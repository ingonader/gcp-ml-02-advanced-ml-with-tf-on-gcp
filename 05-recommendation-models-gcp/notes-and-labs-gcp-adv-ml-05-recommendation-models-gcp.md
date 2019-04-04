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



# Lab 2: [ML on GCP C10] Content-Based Filtering using Neural Networks

1 hour 30 minutes

1 Credit





Rate Lab

## Overview

This lab shows you how to do content-based filtering using DNNs in TensorFlow.

### Objectives

In this lab, you learn to perform the following tasks:

* Build the feature columns for a DNN model using `tf.feature_column`
* Create custom evaluation metrics and add them to TensorBoard
* Train a DNN model for content-based recommendations and perform predictions with that model

## Introduction

In this lab, you'll be providing building a DNN to make content-based recommendations. Content based filtering uses features of the items and users to generate recommendations. In this example, we'll be making recommendations for users on the Kurier.at news site. To do this, we'll be using features based on the news text, title, author, and recency of the article.

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
Note: Follow the prompts during this process.
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

1. In the Datalab browser, navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 10_recommend > labs > content_based_preproc.ipynb**.
2. Read the commentary, click **Clear | Clear all Cells**, then run the Python snippets (Use **Shift+Enter** to run each piece of code) in the cell, step by step.
3. In the Datalab browser, navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 10_recommend > labs > content_based_using_neural_networks.ipynb**.
4. Read the commentary, click **Clear | Clear all Cells**, then run the Python snippets (Use **Shift+Enter** to run each piece of code) in the cell, step by step.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.



# Lab 3: [ML on GCP C10] Collaborative Filtering on Google Analytics data

1 hour 30 minutes

1 Credit





Rate Lab

## Overview

This lab shows you how to do collaborative filtering with Weighted Alternating Least Squares (WALS) matrix refactorization approach.

### Objectives

In this lab, you learn to perform the following tasks:

* Prepare the user-item matrix for use with WALS
* Train a WALSMatrixFactorization within TensorFlow locally and on Cloud ML Engine
* Visualize the embedding vectors with principal components analysis

## Introduction

In this lab, you'll be providing article recommendations for users based on the Kurier.at data. Recall that collaborative filtering doesn't need to know anything about the content. We are only interested in the user-item matrix which defines their relationships.

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

## Create Storage Bucket

*Duration is 2 min*

Create a bucket using the GCP console:

**Step 1**

In your GCP Console, click on the **Navigation menu** (![menu.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/2f22c727923782d61d44dd691c367624adaf6ddd6c33a5b6be24f43b3c913d3b.png)), and select **Storage**.

**Step 2**

Click on **Create bucket**.

**Step 3**

Choose a Regional bucket and set a unique name (use your project ID because it is unique). Then, click **Create**.

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

**Note:** Follow the prompts during this process.

## Checkout notebook into Cloud Datalab

If necessary, wait for Datalab to finish launching. Datalab is ready when you see a message prompting you to do a "Web Preview".

1. Click on the **Web Preview** icon on the top-right corner of the Cloud Shell ribbon. Click on the **Change port**. Switch to port **8081** using the **Change Preview Port** dialog box, and then click on **Change and Preview**.

   ![ChangePort.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/0fe2e17d4078e43c572498391788db31ddc98129a614c51db9fb90116ba4a142.png)

   ![ChangePreviewPort.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/ff07554423a417f49f859aaa5fe3a7ac6bcfc3d3db9add1ab99018c681d74938.png)

   **Note:** The connection to your Datalab instance remains open for as long as the datalab command is active. If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect bdmlvm**' in your new Cloud Shell.

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

1. In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebook > training-data-analyst > courses > machine_learning > deepdive > 10_recommend > labs > wals.ipynb**.
2. Read the commentary, click **Clear | Clear all Cells**, then run the Python snippets (Use **Shift+Enter** to run each piece of code) in the cell, step by step.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.



# Lab 4: [ML on GCP C10] Neural network hybrid recommendation system on Google Analytics

1 hour 30 minutes

1 Credit





Rate Lab

## Overview

This lab shows you how to create a hybrid recommendation system using a combination of approaches and a neural network.

### Objectives

In this lab, you learn to perform the following tasks:

* Generate content-based features from our content-based recommendor
* Create learned embeddings from our collaborative filtering system based on WALS
* Combine the two different systems within a single deep neural network

## Introduction

In this lab, you'll be providing article recommendations for users based on the Kurier.at data. You'll be combining both the content-based and collaborative filtering systems you've developed in previous labs.

## Setup

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

## Task 1: Launch Cloud Datalab

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

## Task 2: Checkout notebook into Cloud Datalab

If necessary, wait for Datalab to finish launching. Datalab is ready when you see a message prompting you to do a "Web Preview".

1. Click on the **Web Preview** icon on the top-right corner of the Cloud Shell ribbon. Click on the **Change port**. Switch to port **8081** using the **Change Preview Port** dialog box, and then click on **Change and Preview**.

   ![ChangePort.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/0fe2e17d4078e43c572498391788db31ddc98129a614c51db9fb90116ba4a142.png)

   ![ChangePreviewPort.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/ff07554423a417f49f859aaa5fe3a7ac6bcfc3d3db9add1ab99018c681d74938.png)

   Note: The connection to your Datalab instance remains open for as long as the datalab command is active. If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command `datalab connect bdmlvm` in your new Cloud Shell.

To clone the course repo in your datalab instance:

**Step 1**

In Cloud Datalab home page (browser), navigate into **“notebooks”** and add a new notebook using the icon ![notebook.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/fef0cc8c36a1856aa4ca73423f2ba59dde635267437c1253c268f366dfe19899.png) on the top left.

**Step 2**

Rename this notebook as **‘repocheckout’**.

**Step 3**

In the new notebook, enter the following commands in the cell, and click on **Run** (on the top navigation bar) to run the commands:

```
%bash
git clone https://github.com/GoogleCloudPlatform/training-data-analyst
rm -rf training-data-analyst/.git
```

![clone.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/8b8e37999b4b2bda8c5862e6686f5dbca32dc52faa953f195ff94b13bc42357b.png)

**Step 4**

Confirm that you have cloned the repo by going back to Datalab browser, and ensure you see the **training-data-analyst** directory. All the files for all labs throughout this course are available in this directory.

![training-data-analyst.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/559d6a731826258adb62b57ead0a57d6e63f47ba667779736c4a779162afaa95.png)

## Task 3: Open a Datalab notebook

1. In the Datalab browser, navigate to **training-data-analyst > courses > machine_learning > deepdive > 10_recommend > labs > hybrid_recommendations > hybrid_recommendations.ipynb**
2. Read the commentary, **Click Clear | Clear all Cells**, then run the Python snippets (Use **Shift+Enter** to run each piece of code) in the cell, step by step.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.





# Lab 5: [ML on GCP C10] Event-Triggered Data Workflow with Cloud Composer

2 hours

1 Credit





Rate Lab

## Overview

This lab shows you how to create an event-triggered data workflow with Cloud Composer.

### Objectives

In this lab, you learn to perform the following tasks:

* Create a Cloud Function that is triggered on file uploads to a GCS bucket
* Create a Cloud Composer DAG that triggers a Cloud Dataflow job to process the newly uploaded data and export it to BigQuery

## Introduction

Your goal in this lab is to automatically ingest a CSV file into BigQuery once it’s uploaded to a GCS bucket. You will be building an automatically-triggered Cloud Composer workflow via a Cloud Function.

## Setup

For each lab, you get a new GCP project and set of resources for a fixed time at no cost.

1. Make sure you signed into Qwiklabs using an **incognito window**.
2. Note the lab's access time (for example, ![img/time.png](https://cdn.qwiklabs.com/aZQJ4BT7uCmM9XR6BTXgTRP1Hfu1T7q6V%2BcnbdEsbpU%3D) and make sure you can finish in that time block.

There is no pause feature. You can restart if needed, but you have to start at the beginning.

1. When ready, click ![img/start_lab.png](https://cdn.qwiklabs.com/XE8x7uvQokyubNwnYKKc%2BvBBNrMlo5iNZiDDzQQ3Ddo%3D).
2. Note your lab credentials. You will use them to sign in to Cloud Platform Console. ![img/open_google_console.png](https://cdn.qwiklabs.com/0d78dhX6IVMVWmixCPPSBbmi5O2GPokCXf1Ps1AkTgI%3D)
3. Click **Open Google Console**.
4. Click **Use another account** and copy/paste credentials for **this** lab into the prompts.

If you use other credentials, you'll get errors or **incur charges**.

1. Accept the terms and skip the recovery resource page.

Do not click **End Lab** unless you are finished with the lab or want to restart it. This clears your work and removes the project.

### Activate Google Cloud Shell

Google Cloud Shell provides command-line access to your GCP resources.

From the GCP Console click the **Cloud Shell** icon on the top right toolbar:

![Cloud Shell Icon](https://cdn.qwiklabs.com/cYAp3uDlYsYcFFNsWmNqW64O9RNumGO5gWDR4GEjkIo%3D)

Then click **START CLOUD SHELL**:

![Start Cloud Shell](https://cdn.qwiklabs.com/%2FrXqdLSk9t%2BseADznDVQNk7Xozp6sXtutHyrPmXDOxM%3D)

You can click **START CLOUD SHELL** immediately when the dialog comes up instead of waiting in the dialog until the Cloud Shell provisions.

It takes a few moments to provision and connects to the environment:

![Cloud Shell Terminal](https://cdn.qwiklabs.com/Ed7y6PTP1vFZDz%2FYJdRWZlhQHKh%2BHV0VUqoXM5BQwZQ%3D)

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

1. Pick a zone in a geographically closeby zone.
2. In **Cloud Shell**, type:

```
datalab create bdmlvm --zone <ZONE>
```

Datalab will take about 5 minutes to start.

**Note**: Follow the prompts during this process.

## Create a Cloud Composer instance

1. In the **Navigation Menu** under Big Data select **Composer**.
2. Select **Create**.
3. Name your instance **mlcomposer**.
4. Select a **Location** nearest you.
5. Leave the other options at their defaults.
6. Click **Create** and continue with the remaining tasks.

Cloud Composer runs on GKE and takes 15-20 minutes to initialize for the first time.

## Grant blob permissions to your service account

1. Click on **Navigation menu > IAM & admin**.
2. Click on pencil icon for `<your-project-id>@appspot.gserviceaccount.com`.
3. Click **Add Another Role** and select **Service Accounts > Service Account Token Creator** role.
4. Click **Save**.

## Checkout notebook into Cloud Datalab

If necessary, wait for Datalab to finish launching. Datalab is ready when you see a message prompting you to do a "Web Preview".

1. Click on the **Web Preview** icon on the top-right corner of the Cloud Shell ribbon. Click on the **Change port**. Switch to port **8081** using the **Change Preview Port** dialog box, and then click on **Change and Preview**.

   ![ChangePort.png](https://cdn.qwiklabs.com/D%2BLhfUB45DxXJJg5F4jbMd3JgSmmFMUdufuQEWukoUI%3D)

   ![ChangePreviewPort.png](https://cdn.qwiklabs.com/%2FwdVRCOkF%2FSfhZqqX%2BOnrGvPw9Pbmt0auZAYxoHXSTg%3D)

**Note:** The connection to your Datalab instance remains open for as long as the datalab command is active. If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command **`datalab connect bdmlvm`** in your new Cloud Shell.

To clone the course repo in your datalab instance:

**Step 1**

In Cloud Datalab home page (browser), navigate into **notebooks** and add a new notebook using the icon ![notebook.png](https://cdn.qwiklabs.com/%2FvDMjDahhWqkynNCPyulnd5jUmdDfBJTwmjzZt%2FhmJk%3D) on the top left.

**Step 2**

Rename this notebook as **repocheckout**.

**Step 3**

In the new notebook, enter the following commands in the cell, and click on **Run** (on the top navigation bar) to run the commands:

```
%bash
git clone https://github.com/GoogleCloudPlatform/training-data-analyst
rm -rf training-data-analyst/.git
```

![clone.png](https://cdn.qwiklabs.com/CjXZ6jeuWQjYk3mhQ8T81ikqbSmBn9NLwJeuF%2FIb2HU%3D)

**Step 4**

Confirm that you have cloned the repo by going back to Datalab browser, and ensure you see the **training-data-analyst** directory. All the files for all labs throughout this course are available in this directory.

![datalab.png](https://cdn.qwiklabs.com/ghxSloBYf7yTE1o%2B3eIkpSOqB9DDigfPeWfxPQgrfw4%3D)

## Open a Datalab notebook

1. In Cloud Datalab, click on the Home icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 10_recommend > labs > composer_gcf_trigger > composertriggered.ipynb**.
2. Read the commentary, click **Clear | Clear all Cells**, then run the Python snippets (Use **Shift+Enter** to run each piece of code) in the cell, step by step.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.



# Lab 6: [ML on GCP C10] End to End Recommendation System

2 hours

1 Credit





Rate Lab

## Overview

This lab shows you how to create an end-to-end recommendation system solution with Cloud Composer.

### Objectives

In this lab, you learn to perform the following tasks:

* Create and deploy your model to App Engine for serving
* Create Cloud Composer and Apache Airflow environments to automatically retrain and redeploy your recommendation model

## Introduction

In this lab, you'll be producing an end-to-end recommendation system based on the previous labs. You'll be orchestrating the system, from data ingest, training, and operationalization with Cloud Composer.

## Setup

For each lab, you get a new GCP project and set of resources for a fixed time at no cost.

1. Make sure you signed into Qwiklabs using an **incognito window**.
2. Note the lab's access time (for example, ![img/time.png](https://cdn.qwiklabs.com/aZQJ4BT7uCmM9XR6BTXgTRP1Hfu1T7q6V%2BcnbdEsbpU%3D) and make sure you can finish in that time block.

There is no pause feature. You can restart if needed, but you have to start at the beginning.

1. When ready, click ![img/start_lab.png](https://cdn.qwiklabs.com/XE8x7uvQokyubNwnYKKc%2BvBBNrMlo5iNZiDDzQQ3Ddo%3D).
2. Note your lab credentials. You will use them to sign in to Cloud Platform Console. ![img/open_google_console.png](https://cdn.qwiklabs.com/0d78dhX6IVMVWmixCPPSBbmi5O2GPokCXf1Ps1AkTgI%3D)
3. Click **Open Google Console**.
4. Click **Use another account** and copy/paste credentials for **this** lab into the prompts.

If you use other credentials, you'll get errors or **incur charges**.

1. Accept the terms and skip the recovery resource page.

Do not click **End Lab** unless you are finished with the lab or want to restart it. This clears your work and removes the project.

### Activate Google Cloud Shell

Google Cloud Shell provides command-line access to your GCP resources.

From the GCP Console click the **Cloud Shell** icon on the top right toolbar:

![Cloud Shell Icon](https://cdn.qwiklabs.com/cYAp3uDlYsYcFFNsWmNqW64O9RNumGO5gWDR4GEjkIo%3D)

Then click **START CLOUD SHELL**:

![Start Cloud Shell](https://cdn.qwiklabs.com/%2FrXqdLSk9t%2BseADznDVQNk7Xozp6sXtutHyrPmXDOxM%3D)

You can click **START CLOUD SHELL** immediately when the dialog comes up instead of waiting in the dialog until the Cloud Shell provisions.

It takes a few moments to provision and connects to the environment:

![Cloud Shell Terminal](https://cdn.qwiklabs.com/Ed7y6PTP1vFZDz%2FYJdRWZlhQHKh%2BHV0VUqoXM5BQwZQ%3D)

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

## Create Storage Bucket

*Duration is 2 min*

Create a bucket using the GCP console:

**Step 1**

In your GCP Console, click on the **Navigation menu** (![menu.png](https://cdn.qwiklabs.com/LyLHJ5I3gtYdRN1pHDZ2JK2vbd1sM6W2viT0OzyRPTs%3D)), and select **Storage**.

**Step 2**

Click on **Create bucket**.

**Step 3**

Choose a Regional bucket and set a unique name (use your project ID because it is unique). Then, click **Create**.

## Launch Cloud Datalab

To launch Cloud Datalab:

1. In **Cloud Shell**, type:

```
gcloud compute zones list
```

1. Pick a zone in a geographically closeby zone.
2. In **Cloud Shell**, type:

```
datalab create bdmlvm --zone <ZONE>
```

Datalab will take about 5 minutes to start.

```
Note: follow the prompts during this process.
```

## Create a Cloud Composer instance

1. In the **Navigation menu** under Big Data select **Composer**.
2. Select **Create**.
3. Name your instance **mlcomposer**.
4. Select a **Location** and **Zone** nearest you.
5. Leave the other options at their defaults.
6. Click **Create** and continue with the remaining tasks.

Cloud Composer runs on GKE and takes 15-20 minutes to initialize for the first time.

## Create a Google App Engine instance

1. Open a second **Cloud Shell** tab (Datalab is the first)
2. In **Cloud Shell**, type:

```
gcloud app regions list
```

1. Pick a region in a geographically closeby.
2. In **Cloud Shell**, type:

```
gcloud app create --region <REGION>
gcloud app create --region europe-west
```

1. In **Cloud Shell**, type:

```
gcloud app update --no-split-health-checks
```

Continue with the remaining tasks while your app engine instance is created.

## Checkout notebook into Cloud Datalab

If necessary, wait for Datalab to finish launching. Datalab is ready when you see a message prompting you to do a "Web Preview".

1. Click on the **Web Preview** icon on the top-right corner of the Cloud Shell ribbon. Click on the **Change port**. Switch to port **8081** using the **Change Preview Port** dialog box, and then click on **Change and Preview**.

   ![ChangePort.png](https://cdn.qwiklabs.com/D%2BLhfUB45DxXJJg5F4jbMd3JgSmmFMUdufuQEWukoUI%3D)

   ![ChangePreviewPort.png](https://cdn.qwiklabs.com/%2FwdVRCOkF%2FSfhZqqX%2BOnrGvPw9Pbmt0auZAYxoHXSTg%3D)

   **Note**: The connection to your Datalab instance remains open for as long as the datalab command is active. If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command `datalab connect bdmlvm` in your new Cloud Shell.

To clone the course repo in your datalab instance:

**Step 1**

In Cloud Datalab home page (browser), navigate into **notebooks** and add a new notebook using the icon ![notebook.png](https://cdn.qwiklabs.com/%2FvDMjDahhWqkynNCPyulnd5jUmdDfBJTwmjzZt%2FhmJk%3D) on the top left.

**Step 2**

Rename this notebook as **repocheckout**.

**Step 3**

In the new notebook, enter the following commands in the cell, and click on **Run** (on the top navigation bar) to run the commands:

```
%bash
git clone https://github.com/GoogleCloudPlatform/training-data-analyst
rm -rf training-data-analyst/.git
```

![clone.png](https://cdn.qwiklabs.com/CjXZ6jeuWQjYk3mhQ8T81ikqbSmBn9NLwJeuF%2FIb2HU%3D)

**Step 4**

Confirm that you have cloned the repo by going back to Datalab browser, and ensure you see the **training-data-analyst** directory. All the files for all labs throughout this course are available in this directory.

![datalab.png](https://cdn.qwiklabs.com/ghxSloBYf7yTE1o%2B3eIkpSOqB9DDigfPeWfxPQgrfw4%3D)

## Open a Datalab notebook

1. In the Datalab browser, navigate to **training-data-analyst > courses > machine_learning > deepdive > 10_recommend > labs > endtoend > endtoend.ipynb**.
2. Read the commentary, Click **Clear | Clear all Cells**, then run the Python snippets (Use **Shift+Enter** to run each piece of code) in the cell, step by step.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.











# [[eof]]

[[todo]]:

* make python and html files of all **notebooks** in all courses!
* git commit all local changes (locally saved notebooks, like the html versions above)
* make pdfs of all **.md-files i**n all courses! (add to .gitignore)
* make "final" model with all bits and pieces:
  * tensorboard infos that work
  * get model from cloud storage to local disk
  * load model from disk and make predictions
  * get "local" evaluation metric

