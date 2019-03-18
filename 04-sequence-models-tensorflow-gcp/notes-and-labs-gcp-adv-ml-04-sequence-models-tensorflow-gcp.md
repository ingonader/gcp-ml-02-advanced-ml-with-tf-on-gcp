# Sequence Models for Time Series and Natural Language Processing with Tensorflow on GCP



## to download materials

software to download materials: <https://github.com/coursera-dl/coursera-dl>

link to coursera material: <https://www.coursera.org/learn/end-to-end-ml-tensorflow-gcp/lecture/wWBkE/effective-ml>

```bash
coursera-dl -u "ingo.nader@gmail.com" sequence-models-tensorflow-gcp -sl en --download-quizzes --download-notebooks --about

#coursera-dl -u "ingo.nader@gmail.com" sequence-models-tensorflow-gcp -sl "en,de" --download-quizzes --download-notebooks --about
```

Done with modified version of coursera-dl: <https://github.com/coursera-dl/coursera-dl/issues/702>



## Notes

* Do labs in Google Chrome (recommended)
* Open Incognito window, go to <http://console.cloud.google.com/>



# Lab 1: [ML on GCP C9] Time Series Prediction with a Linear Model

2 hours

Free





Rate Lab

## Overview

*Duration is 1 min*

In this lab, you will define a linear model to do time series prediction.

### **What you learn**

In this lab, you will learn how to:

* Create a linear model for time series prediction
* Train and serve the model on Cloud Machine Learning Engine

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

**Step 1**

Open Cloud Shell. The Cloud Shell icon is at the top right of the Google Cloud Platform [web console](https://console.cloud.google.com/).

**Step 2**

In Cloud Shell, type:

```
gcloud compute zones list
```

**Note**: Please pick a zone in a geographically close region from the following: **us-east1, us-central1, asia-east1, europe-west1**. These are the regions that currently support Cloud ML Engine jobs. Please verify [here](https://cloud.google.com/ml-engine/docs/tensorflow/environment-overview#cloud_compute_regions) since this list may have changed after this lab was last updated. For example, if you are in the US, you may choose **us-east1-c** as your zone.

**Step 3**

In Cloud Shell, type:

```
datalab create mydatalabvm --zone <ZONE>
```

Replace with a zone name you picked from the previous step.

**Note**: follow the prompts during this process.

Datalab will take about 5 minutes to start.

**Step 4**

Look back at Cloud Shell and follow any prompts. If asked for an ssh passphrase, hit return (for no passphrase).

**Step 5**

If necessary, wait for Datalab to finishing launching. Datalab is ready when you see a message prompting you to do a **Web Preview**.

**Step 6**

Click on **Web Preview** icon on the top-right corner of the Cloud Shell ribbon. Click **Change Port** and enter the port **8081** and click **Change and Preview**.

![web-preview](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/6332967bdbfead14213237528b4e612f00691e996d73e01fe0fec0bd30a8247f.png)

![web-preview](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/083b75223bee4818c3b7c8624dab552a6873de2a5ae58f077a90919940ac134a.png)

![change-port](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/d42795d791a43d871c59c8ad8eccb2290e93209db19e069c02672656761ebbe8.png)

**Note**: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command **datalab connect mydatalabvm** in your new Cloud Shell.

## Clone course repo within your Datalab instance

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

## Time Series Prediction with a Sequence Model

*Duration is 15 min*

The model code is packaged as a separate python module. You will first complete the model code and then switch to the notebook to set some parameters and run the training job.

**Step 1**

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 09_sequence > labs > sinemodel** and open **model.py**.

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

**Step 2**

Scroll down to the function called *linear_model*. Your task is to replace the *#TODO*s in the code, so that this function returns predictions for a linear model.

If you need more help, you may take a look at the complete solution by navigating to : **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 09_sequence > sinemodel** and open **model.py**.

**Step 3**

Now that you have defined your *linear_model*, you are ready to run the training job.

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 09_sequence** and open **sinewaves.ipynb**.

**Step 4**

In Datalab, click on **Clear | Clear all Cells**.

**Step 5**

In the first cell, make sure to replace the project id, bucket and region with your qwiklabs project id, your bucket, and bucket region respectively.

Also, change the --model = *linear* which should be the default.

**Step 6**

Now read the narrative in the following cells and execute each cell in turn.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.



# Lab 2: [ML on GCP C9] Time Series Prediction with a DNN Model

2 hours

Free





Rate Lab

## Overview

*Duration is 1 min*

In this lab, you will define a DNN (Deep Neural Network) model to do time series prediction

### What you learn

In this lab, you will learn how to:

* Create a DNN model for time series prediction
* Train and serve the model on Cloud Machine Learning Engine

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

**Step 1**

Open Cloud Shell. The Cloud Shell icon is at the top right of the Google Cloud Platform [web console](https://console.cloud.google.com/).

**Step 2**

In Cloud Shell, type:

```
gcloud compute zones list
```

**Note**: Please pick a zone in a geographically close region from the following: **us-east1, us-central1, asia-east1, europe-west1**. These are the regions that currently support Cloud ML Engine jobs. Please verify [here](https://cloud.google.com/ml-engine/docs/tensorflow/environment-overview#cloud_compute_regions) since this list may have changed after this lab was last updated. For example, if you are in the US, you may choose **us-east1-c** as your zone.

**Step 3**

In Cloud Shell, type:

```
datalab create mydatalabvm --zone <ZONE>
```

Replace with a zone name you picked from the previous step.

**Note**: follow the prompts during this process.

Datalab will take about 5 minutes to start.

**Step 4**

Look back at Cloud Shell and follow any prompts. If asked for an ssh passphrase, hit return (for no passphrase).

**Step 5**

If necessary, wait for Datalab to finishing launching. Datalab is ready when you see a message prompting you to do a **Web Preview**.

**Step 6**

Click on **Web Preview** icon on the top-right corner of the Cloud Shell ribbon. Click **Change Port** and enter the port **8081** and click **Change and Preview**.

![web-preview](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/6332967bdbfead14213237528b4e612f00691e996d73e01fe0fec0bd30a8247f.png)

![web-preview](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/083b75223bee4818c3b7c8624dab552a6873de2a5ae58f077a90919940ac134a.png)

![change-port](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/d42795d791a43d871c59c8ad8eccb2290e93209db19e069c02672656761ebbe8.png)

**Note**: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command **datalab connect mydatalabvm** in your new Cloud Shell.

## Clone course repo within your Datalab instance

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

## Time Series Prediction with a Sequence Model

*Duration is 15 min*

The model code is packaged as a separate python module. You will first complete the model code and then switch to the notebook to set some parameters and run the training job.

**Step 1**

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 09_sequence > labs > sinemodel** and open **model.py**.

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

**Step 2**

Scroll down to the function called `dnn_model`. Your task is to replace the TODO in the code, so that this function returns predictions for a DNN.

If you need more help, you may take a look at the complete solution by navigating to : **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 09_sequence > sinemodel** and open **model.py**.

**Step 3**

Now that you have defined your DNN model, you are ready to run the training job.

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 09_sequence** and open **sinewaves.ipynb**.

**Step 4**

In Datalab, click on **Clear | Clear all Cells**.

**Step 5**

In the first cell, make sure to replace the project id, bucket and region with your qwiklabs project id, your bucket, and bucket region respectively.

Also, change the **--model = dnn** (linear is the default, be sure to change it).

**Step 6**

Now read the narrative in the following cells and execute each cell in turn.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.

# Lab 3: [ML on GCP C9] Time Series Prediction with a CNN Model

2 hours

Free





Rate Lab

## Overview

*Duration is 1 min*

In this lab, you will define a CNN (Convolutional Neural Network) model to do time series prediction.

### **What you learn**

In this lab, you will learn how to:

* Create a CNN model for time series prediction
* Train and serve the model on Cloud Machine Learning Engine

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

## Create storage bucket

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

**Step 1**

Open Cloud Shell. The Cloud Shell icon is at the top right of the Google Cloud Platform [web console](https://console.cloud.google.com/).

**Step 2**

In Cloud Shell, type:

```
gcloud compute zones list
```

**Note**: Please pick a zone in a geographically close region from the following: **us-east1, us-central1, asia-east1, europe-west1**. These are the regions that currently support Cloud ML Engine jobs. Please verify [here](https://cloud.google.com/ml-engine/docs/tensorflow/environment-overview#cloud_compute_regions) since this list may have changed after this lab was last updated. For example, if you are in the US, you may choose **us-east1-c** as your zone.

**Step 3**

In Cloud Shell, type:

```
datalab create mydatalabvm --zone <ZONE>
```

Replace with a zone name you picked from the previous step.

**Note**: follow the prompts during this process.

Datalab will take about 5 minutes to start.

**Step 4**

Look back at Cloud Shell and follow any prompts. If asked for an ssh passphrase, hit return (for no passphrase).

**Step 5**

If necessary, wait for Datalab to finishing launching. Datalab is ready when you see a message prompting you to do a **Web Preview**.

**Step 6**

Click on **Web Preview** icon on the top-right corner of the Cloud Shell ribbon. Click **Change Port** and enter the port **8081** and click **Change and Preview**.

![web-preview](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/6332967bdbfead14213237528b4e612f00691e996d73e01fe0fec0bd30a8247f.png)

![web-preview](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/083b75223bee4818c3b7c8624dab552a6873de2a5ae58f077a90919940ac134a.png)

![change-port](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/d42795d791a43d871c59c8ad8eccb2290e93209db19e069c02672656761ebbe8.png)

**Note**: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command **datalab connect mydatalabvm** in your new Cloud Shell.

## Clone course repo within your Datalab instance

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

## Time Series Prediction with a Sequence Model

*Duration is 15 min*

The model code is packaged as a separate python module. You will first complete the model code and then switch to the notebook to set some parameters and run the training job.

**Step 1**

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 09_sequence > labs > sinemodel** and open **model.py**.

**Note**: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

**Step 2**

Scroll down to the function called *cnn_model*. Your task is to replace the *TODO* in the code, so that this function returns predictions for a CNN.

If you need more help, you may take a look at the complete solution by navigating to : **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 09_sequence > sinemodel** and open **model.py**

**Step 3**

Now that you have defined your CNN model, you are ready to run the training job.

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 09_sequence** and open **sinewaves.ipynb**.

**Step 4**

In Datalab, click on **Clear | Clear all Cells**.

**Step 5**

In the first cell, make sure to replace the project id, bucket and region with your qwiklabs project id, your bucket, and bucket region respectively.

Also, change the --model = *cnn* (linear is the default, be sure to change it)

**Step 6**

Now read the narrative in the following cells and execute each cell in turn.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.





# Lab 4: [ML on GCP C9] Time Series Prediction with a RNN Model

2 hours

Free





Rate Lab

## Overview

*Duration is 1 min*

In this lab, you will define a RNN (Recurrent Neural Network) model to do time series prediction.

### What you learn

In this lab, you will learn how to:

* Create a RNN model for time series prediction
* Train and serve the model on Cloud Machine Learning Engine

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

## Create storage bucket

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

**Step 1**

Open Cloud Shell. The Cloud Shell icon is at the top right of the Google Cloud Platform [web console](https://console.cloud.google.com/).

**Step 2**

In Cloud Shell, type:

```
gcloud compute zones list
```

**Note**: Please pick a zone in a geographically close region from the following: **us-east1, us-central1, asia-east1, europe-west1**. These are the regions that currently support Cloud ML Engine jobs. Please verify [here](https://cloud.google.com/ml-engine/docs/tensorflow/environment-overview#cloud_compute_regions) since this list may have changed after this lab was last updated. For example, if you are in the US, you may choose **us-east1-c** as your zone.

**Step 3**

In Cloud Shell, type:

```
datalab create mydatalabvm --zone <ZONE>
```

Replace with a zone name you picked from the previous step.

**Note**: follow the prompts during this process.

Datalab will take about 5 minutes to start.

**Step 4**

Look back at Cloud Shell and follow any prompts. If asked for an ssh passphrase, hit return (for no passphrase).

**Step 5**

If necessary, wait for Datalab to finishing launching. Datalab is ready when you see a message prompting you to do a **Web Preview**.

**Step 6**

Click on **Web Preview** icon on the top-right corner of the Cloud Shell ribbon. Click **Change Port** and enter the port **8081** and click **Change and Preview**.

![web-preview](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/6332967bdbfead14213237528b4e612f00691e996d73e01fe0fec0bd30a8247f.png)

![web-preview](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/083b75223bee4818c3b7c8624dab552a6873de2a5ae58f077a90919940ac134a.png)

![change-port](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/d42795d791a43d871c59c8ad8eccb2290e93209db19e069c02672656761ebbe8.png)

**Note**: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command **datalab connect mydatalabvm** in your new Cloud Shell.

## Clone course repo within your Datalab instance

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

## Time Series Prediction with a Sequence Model

*Duration is 15 min*

The model code is packaged as a separate python module. You will first complete the model code and then switch to the notebook to set some parameters and run the training job.

**Step 1**

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 09_sequence > labs > sinemodel** and open **model.py**.

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

**Step 2**

Scroll down to the function called `rnn_model`. Your task is to replace the TODO in the code, so that this function returns predictions for an RNN. To construct your model, make use of either the [LSTMCell](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/LSTMCell) or [GRUCell](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/GRUCell)classes and the [dynamic_rnn](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn) function.

If you need more help, you may take a look at the complete solution by navigating to : **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 09_sequence > sinemodel** and open **model.py**

**Step 3**

Now that you have defined your RNN model, you are ready to run the training job.

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 09_sequence** and open **sinewaves.ipynb**.

**Step 4**

In Datalab, click on **Clear | Clear all Cells**.

**Step 5**

In the first cell, make sure to replace the project id, bucket and region with your qwiklabs project id, your bucket, and bucket region respectively.

Also, change the --model = *rnn* (linear is the default, be sure to change it)

**Step 6**

Now read the narrative in the following cells and execute each cell in turn.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.



# Lab 5: [ML on GCP C9] Time Series Prediction with a Two-Layer RNN Model

2 hours

Free





Rate Lab

## Overview

*Duration is 1 min*

In this lab, you will define a Two-Layer RNN (Recurrent Neural Network) model to do time series prediction.

### **What you learn**

In this lab, you will learn how to:

* Create a Two-Layer RNN model for time series prediction
* Train and serve the model on Cloud Machine Learning Engine

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

## Create storage bucket

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

**Step 1**

Open Cloud Shell. The Cloud Shell icon is at the top right of the Google Cloud Platform [web console](https://console.cloud.google.com/).

**Step 2**

In Cloud Shell, type:

```
gcloud compute zones list
```

**Note**: Please pick a zone in a geographically close region from the following: **us-east1, us-central1, asia-east1, europe-west1**. These are the regions that currently support Cloud ML Engine jobs. Please verify [here](https://cloud.google.com/ml-engine/docs/tensorflow/environment-overview#cloud_compute_regions) since this list may have changed after this lab was last updated. For example, if you are in the US, you may choose **us-east1-c** as your zone.

**Step 3**

In Cloud Shell, type:

```
datalab create mydatalabvm --zone <ZONE>
```

Replace with a zone name you picked from the previous step.

**Note**: follow the prompts during this process.

Datalab will take about 5 minutes to start.

**Step 4**

Look back at Cloud Shell and follow any prompts. If asked for an ssh passphrase, hit return (for no passphrase).

**Step 5**

If necessary, wait for Datalab to finishing launching. Datalab is ready when you see a message prompting you to do a **Web Preview**.

**Step 6**

Click on **Web Preview** icon on the top-right corner of the Cloud Shell ribbon. Click **Change Port** and enter the port **8081** and click **Change and Preview**.

![web-preview](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/6332967bdbfead14213237528b4e612f00691e996d73e01fe0fec0bd30a8247f.png)

![web-preview](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/083b75223bee4818c3b7c8624dab552a6873de2a5ae58f077a90919940ac134a.png)

![change-port](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/d42795d791a43d871c59c8ad8eccb2290e93209db19e069c02672656761ebbe8.png)

**Note**: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command **datalab connect mydatalabvm** in your new Cloud Shell.

## Clone course repo within your Datalab instance

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

## Time Series Prediction with a Sequence Model

*Duration is 15 min*

The model code is packaged as a separate python module. You will first complete the model code and then switch to the notebook to set some parameters and run the training job.

**Step 1**

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 09_sequence > labs > sinemodel** and open **model.py**.

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

**Step 2**

Scroll down to the function called *rnn2_model*. Your task is to replace the *TODO* in the code, so that this function returns predictions for an RNN with multiple layers. To construct your model, make use of either the[LSTMCell](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/LSTMCell) or [GRUCell](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/GRUCell) classes, the [MultiRNNCell](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/MultiRNNCell) function, and the[dynamic_rnn](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn) function.

If you need more help, you may take a look at the complete solution by navigating to : **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 09_sequence > sinemodel** and open **model.py**

**Step 3**

Now that you have defined your Two-Layer RNN model, you are ready to run the training job.

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 09_sequence** and open **sinewaves.ipynb**.

**Step 4**

In Datalab, click on **Clear | Clear all Cells**.

**Step 5**

In the first cell, make sure to replace the project id, bucket and region with your qwiklabs project id, your bucket, and bucket region respectively.

Also, change the --model = *rnn2* (linear is the default, be sure to change it)

**Step 6**

Now read the narrative in the following cells and execute each cell in turn.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.



# Lab 6: [ML on GCP C9] An RNN Model for Temperature Data

2 hours

Free





Rate Lab

## Overview

*Duration is 1 min*

In this lab, you will define a RNN (Recurrent Neural Network) model to do time-series prediction on real world temperature data.

### What you learn

In this lab, you will learn how to:

* Run the RNN model with basic settings as a baseline
* Adjust hyperparameters for performance tuning
* Adjust the window of predictions for less noisy data

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

**Step 1**

Open Cloud Shell. The Cloud Shell icon is at the top right of the Google Cloud Platform [web console](https://console.cloud.google.com/).

**Step 2**

In Cloud Shell, type:

```
gcloud compute zones list
```

**Note**: Please pick a zone in a geographically close region from the following: **us-east1, us-central1, asia-east1, europe-west1**. These are the regions that currently support Cloud ML Engine jobs. Please verify [here](https://cloud.google.com/ml-engine/docs/tensorflow/environment-overview#cloud_compute_regions) since this list may have changed after this lab was last updated. For example, if you are in the US, you may choose **us-east1-c** as your zone.

**Step 3**

In Cloud Shell, type:

```
datalab create mydatalabvm --zone <ZONE>
```

Replace with a zone name you picked from the previous step.

**Note**: follow the prompts during this process.

Datalab will take about 5 minutes to start.

**Step 4**

Look back at Cloud Shell and follow any prompts. If asked for an ssh passphrase, hit return (for no passphrase).

**Step 5**

If necessary, wait for Datalab to finishing launching. Datalab is ready when you see a message prompting you to do a **Web Preview**.

**Step 6**

Click on **Web Preview** icon on the top-right corner of the Cloud Shell ribbon. Click **Change Port** and enter the port **8081** and click **Change and Preview**.

![web-preview](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/6332967bdbfead14213237528b4e612f00691e996d73e01fe0fec0bd30a8247f.png)

![web-preview](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/083b75223bee4818c3b7c8624dab552a6873de2a5ae58f077a90919940ac134a.png)

![change-port](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/d42795d791a43d871c59c8ad8eccb2290e93209db19e069c02672656761ebbe8.png)

**Note**: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command **datalab connect mydatalabvm** in your new Cloud Shell.

## Clone course repo within your Datalab instance

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

## Time Series Weather Data Prediction

*Duration is 15 min*

**Step 1**

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 09_sequence > labs** and open **temperatures.ipynb**.

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

**Step 2**

In Datalab, click on **Clear | Clear all Cells** (click on Clear, then in the drop-down menu, select Clear all Cells).

Read through the assignment steps required in the first notebook cell (starting with *Run the notebook as it is. Look at the data visualisations)* and complete them in your notebook.

If you need more help, you may take a look at the complete solution by navigating to : **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 09_sequence** and open **temperatures.ipynb**

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.



# Lab 7: [ML on GCP C9] Text Classification using TensorFlow/Keras on Cloud ML Engine

2 hours

Free





Rate Lab

## Overview

*Duration is 1 min*

In this lab, you will define a text classification model to look at the titles of articles and figure out whether the article came from the New York Times, TechCrunch or GitHub.

### **What you learn**

In this lab, you will learn how to:

* Creating datasets for Machine Learning using BigQuery
* Creating a text classification model using the Estimator API with a Keras model
* Training on Cloud ML Engine
* Deploying the model
* Predicting with model
* Rerun with pre-trained embedding

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

**Step 1**

Open Cloud Shell. The Cloud Shell icon is at the top right of the Google Cloud Platform [web console](https://console.cloud.google.com/).

**Step 2**

In Cloud Shell, type:

```
gcloud compute zones list
```

**Note**: Please pick a zone in a geographically close region from the following: **us-east1, us-central1, asia-east1, europe-west1**. These are the regions that currently support Cloud ML Engine jobs. Please verify [here](https://cloud.google.com/ml-engine/docs/tensorflow/environment-overview#cloud_compute_regions) since this list may have changed after this lab was last updated. For example, if you are in the US, you may choose **us-east1-c** as your zone.

**Step 3**

In Cloud Shell, type:

```
datalab create mydatalabvm --zone <ZONE>
```

Replace with a zone name you picked from the previous step.

**Note**: follow the prompts during this process.

Datalab will take about 5 minutes to start.

**Step 4**

Look back at Cloud Shell and follow any prompts. If asked for an ssh passphrase, hit return (for no passphrase).

**Step 5**

If necessary, wait for Datalab to finishing launching. Datalab is ready when you see a message prompting you to do a **Web Preview**.

**Step 6**

Click on **Web Preview** icon on the top-right corner of the Cloud Shell ribbon. Click **Change Port** and enter the port **8081** and click **Change and Preview**.

![web-preview](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/6332967bdbfead14213237528b4e612f00691e996d73e01fe0fec0bd30a8247f.png)

![web-preview](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/083b75223bee4818c3b7c8624dab552a6873de2a5ae58f077a90919940ac134a.png)

![change-port](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/d42795d791a43d871c59c8ad8eccb2290e93209db19e069c02672656761ebbe8.png)

**Note**: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command **datalab connect mydatalabvm** in your new Cloud Shell.

## Clone course repo within your Datalab instance

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

## Building a Sequence Model for Text Classification

*Duration is 15 min*

The model code is packaged as a separate python module. You will first complete the model code and then switch to the notebook to set some parameters and run the training job.

**Step 1**

In Cloud Datalab, click on the Home icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 09_sequence > labs** and open **text_classification.ipynb**.

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

**Step 2**

Read through the assignment steps required in the first notebook cell and complete them in your notebook.

Be sure to complete the *#TODO*s in the companion model.py notebook found in **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 09_sequence > labs > txtclsmodel > trainer > model.py**.

If you need more help, you may take a look at the complete solution by navigating **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 09_sequence > txtclsmodel > trainer > model.py**.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.



# Lab 8: [ML on GCP C9] Using pre-trained embeddings with TensorFlow Hub

2 hours

Free





Rate Lab

## Overview

*Duration is 1 min*

In this lab, you will build a model using pre-trained embeddings from TensorFlow hub.

TensorFlow Hub is a library for the publication, discovery, and consumption of reusable parts of machine learning models. A module is a self-contained piece of a TensorFlow graph, along with its weights and assets, that can be reused across different tasks in a process known as transfer learning, which we covered as part of the course on Image Models.

### **What you learn**

In this lab, you will learn how to:

* How to instantiate a TensorFlow Hub module
* How to find pre-trained TensorFlow Hub modules for a variety of purposes
* How to examine the embeddings of a Hub module
* How one Hub module composes representations of sentences from individual words
* How to assess word embeddings using a semantic similarity test

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

## Create storage bucket

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

**Step 1**

Open Cloud Shell. The Cloud Shell icon is at the top right of the Google Cloud Platform [web console](https://console.cloud.google.com/).

**Step 2**

In Cloud Shell, type:

```
gcloud compute zones list
```

**Note**: Please pick a zone in a geographically close region from the following: **us-east1, us-central1, asia-east1, europe-west1**. These are the regions that currently support Cloud ML Engine jobs. Please verify [here](https://cloud.google.com/ml-engine/docs/tensorflow/environment-overview#cloud_compute_regions) since this list may have changed after this lab was last updated. For example, if you are in the US, you may choose **us-east1-c** as your zone.

**Step 3**

In Cloud Shell, type:

```
datalab create mydatalabvm --zone <ZONE>
```

Replace with a zone name you picked from the previous step.

**Note**: follow the prompts during this process.

Datalab will take about 5 minutes to start.

**Step 4**

Look back at Cloud Shell and follow any prompts. If asked for an ssh passphrase, hit return (for no passphrase).

**Step 5**

If necessary, wait for Datalab to finishing launching. Datalab is ready when you see a message prompting you to do a **Web Preview**.

**Step 6**

Click on **Web Preview** icon on the top-right corner of the Cloud Shell ribbon. Click **Change Port** and enter the port **8081** and click **Change and Preview**.

![web-preview](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/6332967bdbfead14213237528b4e612f00691e996d73e01fe0fec0bd30a8247f.png)

![web-preview](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/083b75223bee4818c3b7c8624dab552a6873de2a5ae58f077a90919940ac134a.png)

![change-port](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/d42795d791a43d871c59c8ad8eccb2290e93209db19e069c02672656761ebbe8.png)

**Note**: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command **datalab connect mydatalabvm** in your new Cloud Shell.

## Clone course repo within your Datalab instance

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

## Using pre-trained embeddings with TensorFlow Hub

*Duration is 15 min*

**Step 1**

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 09_sequence > labs** and open **reusable-embeddings.ipynb**.

**Note**: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

**Step 2**

Read through the assignment steps required in the first notebook cell and complete them in your notebook.

If you need more help, you may take a look at the complete solution by navigating to : **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 09_sequence** and open **reusable-embeddings.ipynb**.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.



# Lab 9: [ML on GCP C9] Text generation using tensor2tensor on Cloud ML Engine

2 hours

Free





Rate Lab

## Overview

*Duration is 1 min*

This notebook illustrates using the tensor2tensor library to do from-scratch, distributed training of a poetry model. Then, the trained model is used to complete new poems.

### What you learn

In this lab, you will learn how to:

* Create a training dataset from text data
* Utilize the tensor2tensor library for text classification
* Train the model locally
* Train on Cloud Machine Learning Engine

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

## Create storage bucket

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

**Step 1**

Open Cloud Shell. The Cloud Shell icon is at the top right of the Google Cloud Platform [web console](https://console.cloud.google.com/).

**Step 2**

In Cloud Shell, type:

```
gcloud compute zones list
```

**Note**: Please pick a zone in a geographically close region from the following: **us-east1, us-central1, asia-east1, europe-west1**. These are the regions that currently support Cloud ML Engine jobs. Please verify [here](https://cloud.google.com/ml-engine/docs/tensorflow/environment-overview#cloud_compute_regions) since this list may have changed after this lab was last updated. For example, if you are in the US, you may choose **us-east1-c** as your zone.

**Step 3**

In Cloud Shell, type:

```
datalab create mydatalabvm --zone <ZONE>
```

Replace with a zone name you picked from the previous step.

**Note**: follow the prompts during this process.

Datalab will take about 5 minutes to start.

**Step 4**

Look back at Cloud Shell and follow any prompts. If asked for an ssh passphrase, hit return (for no passphrase).

**Step 5**

If necessary, wait for Datalab to finishing launching. Datalab is ready when you see a message prompting you to do a **Web Preview**.

**Step 6**

Click on **Web Preview** icon on the top-right corner of the Cloud Shell ribbon. Click **Change Port** and enter the port **8081** and click **Change and Preview**.

![web-preview](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/6332967bdbfead14213237528b4e612f00691e996d73e01fe0fec0bd30a8247f.png)

![web-preview](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/083b75223bee4818c3b7c8624dab552a6873de2a5ae58f077a90919940ac134a.png)

![change-port](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/d42795d791a43d871c59c8ad8eccb2290e93209db19e069c02672656761ebbe8.png)

**Note**: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command **datalab connect mydatalabvm** in your new Cloud Shell.

## Clone course repo within your Datalab instance

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

## Text generation using tensor2tensor on Cloud ML Engine

*Duration is 15 min*

**Step 1**

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 09_sequence** and open **poetry.ipynb**.

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

**Step 2**

Read through the assignment steps required in the first notebook cell and complete them in your notebook.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.



# Lab 10: [ML on GCP C9] Getting Started with Dialogflow

2 hours

Free





Rate Lab

## Overview

This lab shows you how to build a simple Dialogflow agent, walking you through the most important features of Dialogflow. You'll learn how to:

* [Create a Dialogflow account](https://dialogflow.com/docs/getting-started/create-account) and [your first Dialogflow agent](https://dialogflow.com/docs/getting-started/first-agent), which lets you define a natural language understanding model.

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

## Create your Dialogflow account

This section describes how to create and log in to a Dialogflow account

### Create your Dialogflow account

Now that you're signed into your Qwiklabs student account in an incognito (private browser) window, you can sign into Dialogflow [here](https://console.dialogflow.com/api-client/#/login) by following these steps:

1. Click **Google**. ![6aecd425e66e8198.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/2c1468a64914ab8c7bdd45e162b00c0dca4ea89221409516df9fe8aad3e48aa0.png)
2. Be sure to sign in with your Qwiklabs account
3. Allow Dialogflow to access your Google account. [See a list of the permissions and what they're used for](https://dialogflow.com/docs/concepts/permissions).

Lastly, you'll be taken to Dialogflow's terms of service, which you'll need to accept in order to use Dialogflow.

![1a86ef1562a2aae.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/cac2c32025706ff463068154f2d784bfb6338d87af0d8f7cb82359bff88eec70.png)

### Next steps

Next, you'll create your first Dialogflow agent and test it out.

## Create and query your first agent

This section describes how to create and try out your first Dialogflow agent.

**Note:** Before you start, make sure you've [created a Dialogflow account](https://dialogflow.com/docs/getting-started/create-account).

### Create your first Dialogflow agent

To create a Dialogflow agent:

1. Open a browser and [log in to Dialogflow](https://console.dialogflow.com/api-client/#/login).
2. Click **Create agent** in the left menu.

![4767285463efdad7.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/64704472654e88a34ebc4d6e4b7b934c27370dc9dfccf88c8c74d1d1d13c6e64.png)

1. Enter your agent's name, default language, and default time zone, then click the **Create** button. ![32615f1898bb1252.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/2303d4c70d7e1ad4d804535de28ac5d5fb4e8168e91b06c82796b1c75972f283.png)

### The Dialogflow console

![af6f286a86a0aed.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/05afd9b7a7b983e625783715e488557d113b1479f0b72f3cfda29c48bc0515d4.png)

You should now see the Dialogflow console and the menu panel on the left. If you're working on a smaller screen and the menu is hidden, click on the menu button in the upper left corner. The settings button takes you to the current [agent's settings](https://dialogflow.com/docs/agents#settings).

The middle of the page will show the list of intents for the agent. By default, Dialogflow agents start with two intents. Your agent matches the **Default Fallback Intent** when it doesn't understand what your users say. The **Default Welcome Intent** greets your users. These can be altered to customize the experience.

![7b1ddb9e199ae3a2.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/e1988a283897e31c275aeb5c53d73ce5b063973eba8226c9cd058e4700333344.png)

![733bd1bf20f32da1.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/771b977eb48e5eb339d830f242386245ee9beb8b192b7734f7f00545289c16b4.png)

On the right is the Dialogflow simulator. This lets you try out your agent by speaking or typing messages.

### Query your agent

Agents are best described as NLU (Natural Language Understanding) modules. These can be included in your app, product, or service and transform natural user requests into actionable data.

![65ea14ce62e3c036.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/f400d9ac458669294da32573275b2c54a28d059ee54b785c9035787894372553.png)

Time to try out your agent! In the **Dialogflow simulator** on the right, click into the text field that says **Try it now**, type **hi**, and press enter.

You just spoke to your Dialogflow agent! You may notice your agent understood you. Since your input matched to the Default Welcome Intent, you received one of the default replies inside the welcome intent.

In the case that your agent doesn't understand you, the Default Fallback Intent is matched and you receive one of the default replies inside that intent.

The Default Fallback Intent reply prompts the user to reframe their query in terms that can be matched. You can change the responses within the Default Fallback Intent to provide example queries and guide the user to make requests that can match an intent.

### Create your first intent

Dialogflow uses intents to categorize a user's intentions. Intents have **Training Phrases**, which are examples of what a user might say to your agent. For example, someone wanting to know the name of your agent might ask, "What is your name?", "Do you have a name?", or just say "name". All of these queries are unique but have the same intention: to get the name of your agent.

To cover this query, create a "name" intent:

1. Click on the plus add next to **Intents** in the left menu.
2. Add the name "name" into the **Intent name** text field.
3. In the **Training Phrases** section, click **Add Training Phrases** enter the following, pressing enter after each entry:

* What is your name?
* Do you have a name?
* name

1. In the **Responses** section, click **Add Response** enter the following response:

* My name is Dialogflow!

![fd91a8fc36247980.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/c98fed6177e83363ff2c6a7555d5a28c0d4ceb0ac96528c1e10b31d074ce677a.png)5. Click the **Save** button.

#### Try it out!

![e729697de0f2c154.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/000c936686e68632ba35587f55067fc6402a62a6a2d81c2b9ef551c9132b0138.png)

Now try asking your agent for its name. In the simulator on the right, type "What's your name?" and press enter.

Your agent now responds to the query correctly. Notice that even though your query was a little different from the training phrase ("What's your name?" versus "What is your name?"), Dialogflow still matched the query to the right intent.

Dialogflow uses training phrases as examples for a machine learning model to match users' queries to the correct intent. The [machine learning](https://dialogflow.com/docs/agents/machine-learning)model checks the query against every intent in the agent, gives every intent a score, and the highest-scoring intent is matched. If the highest scoring intent has a very low score, the fallback intent is matched.

## Extract data with entities

This section describes how to extract data from a user's query.

### Add parameters to your intents

Parameters are important and relevant words or phrases in a user's query that are extracted so your agent can provide a proper response. You'll create a new intent with parameters for spoken and programming languages to explore how these can match specific intents and be included in your responses.

1. Create a new intent by clicking on the plus add next to **Intents** in the left menu.
2. Name the intent "Languages" at the top of the intent page.
3. Add the following as Training phrases:

* I know English
* I speak French
* I know how to write in German

![9f99c39716a71508.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/a0ef527192cc4702a91b53cdfc4a7ce1206f74d74326246e04af7b92bf1b7553.png)

Dialogflow automatically detects known parameters in your Training phrases and creates them for you.

Below the **Training phrases** section, Dialogflow fills out the parameter table with the information it gathered:

* The parameter is optional (not required)
* named language
* corresponds to the [system entity](https://dialogflow.com/docs/entities#system_entities) type [@sys.language](https://dialogflow.com/docs/reference/system-entities#other)
* has the value of $language
* is not a [list](https://dialogflow.com/docs/actions-and-parameters#is_list)

**Note:** If entities aren't automatically detected, you can highlight the text in the Training phrase and [manually annotate the entity](https://dialogflow.com/docs/intents#manual_annotation).

### Use parameter data

![1e24fd4a42bdaf08.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/0a5429614c77cba88e68f92bd86aab9d317b7f00de1fa6f6e17020272d1baf35.png)

The value of a parameter can be used in your responses. In this case, you can use $language in your responses and it will be replaced with the language specified in the query to your agent.

1. In the **Responses** section, add the following response and click the **Save**button:

* Wow! I didn't know you knew $language

#### Try it out!

![626007876cbc6424.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/6970e66f676ebec5ca3269e387f13fa140c38ce2fb460f41c04cc3da8c1c35ff.png)

Now, query your agent with "I know Russian" in the simulator in the right panel.

You can see in the bottom of the simulator output that Dialogflow correctly extracted the language parameter with the value "Russian" from the query. In the response, you can see "Russian" was correctly inserted where the parameter value was used.

### Create your own entities

You can also create your own entities, which function similarly to Dialogflow's system entities.

To create an entity:

1. Click on the plus add next to **Entities** in the left menu.
2. Enter "ProgrammingLanguage" for the name of the entity.
3. Click on the text field and add the following entries:

* JavaScript
* Java
* Python

1. When you enter an entry, pressing tab moves your cursor into the synonym field. Add the following synonyms for each entry: ![7f2ce5ab86b464bc.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/7930be0ef9a1940ba7b778fa2343772ab7fea3a9bd0a729b544bfab79a60d000.png)Click the **Save** button.

Each entity type has to have the following:

* a name to define the category (ProgrammingLanguage)
* one or more entries (JavaScript)
* one or more synonyms (js, JavaScript)

Dialogflow can handle simple things like plurality and capitalization, but make sure to add all possible synonyms for your entries. The more you add, the better your agent can determine your entities.

### Add your new entities

Now that we've defined our entity for programming languages, add Training Phrases to the "Languages" intent:

1. Click on **Intents** in the left menu, and then click on the "Languages" intent.
2. Add the following as Training phrases: ![bdf19643e9a83283.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/8c3bf348e7d3e772c437f138910950bfc7878115c43eca19d942dba07887c404.png)

* I know javascript
* I know how to code in Java

1. You should see the programming languages automatically annotated in the Training phrases you entered. This adds the ProgrammingLanguage parameter to the table, which is below the **Training phrases** section.![cfc3cdc2079f9774.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/aa2dae1660930c7f2db270ecb93f9a744ac39778b3c1db5b254f73cd8214a1e8.png)![98144d1b98cb37b2.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/a77bc3f6e6c6648cf4f293600308394581caba54449ae14427c420fb50fa8b04.png)
2. In the **Responses** section, add "$ProgrammingLanguage is cool" and then click the **Save** button.

#### Try it out!

![3cfb5275b7696c13.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/f256d17bd929a84b4a0e132457e9a1bb9d2f4d4e54913981798cfbc9abe0333f.png)

## Manage state with contexts

This section describes how to track conversational states with follow-up intents and contexts.

### Add contexts to conversational state

1. Click on **Intents** in the left menu, and then click on the "Languages" intent.
2. Extend one of the original Text response in the **Response** section to the following:

* Wow! I didn't know you knew `$language`. How long have you known `$language`?

![c9ed5bf4ba507451.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/3b7f03cf766299e598810b453ff98ce77f5758a7e577d6c1492bddc28a69b43b.png)

1. Click the **Save** button.
2. Click on **Intents** in the left menu.
3. Hover over the "Languages" intent and click on **Add follow-up intent**:

![cfdcd099cbf30ee9.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/7b9d3e9d4028c9f91763fd36bd98d866607768ed13edf1cb6c99d08a033f31b4.png)

1. Click on **Custom** in the revealed list:![f12f2a0d459fc28.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/046d0d51b1a1fc9d33a37bf2bed3c3417014416daa57e14d4778b4051d1c42bf.png)

Dialogflow automatically names the follow-up intent "Languages - custom", and the arrow indicates the relationship between the intents.

![67b4fd1300f4adb8.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/21b66d317f713bc5300979a1b6ba8bc513e8b0218a0c67d87b08ba725da0c292.png)

### Intent matching with follow-up intents

Follow-up intents are only matched after the parent intent has been matched. Since this intent is only matched after the "Languages" intent, we can assume that the user has just been asked the question "How long have you known $language?". You'll now add Training Phrases indicating users' likely answers to that question.

![38e74f1867ee4a6a.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/249b68a7e204a609f732cf4f54a9a350eb79893b6ad9e0439de4885d978a71e1.png)

1. Click on **Intents** in the left menu and then click on the "Languages - custom" intent.
2. Add the following Training Phrases:

* 3 years
* about 4 days
* for 5 years

1. Click the **Save** button.

### Try it out

Try this out in the **Dialogflow simulator** on the right. First, match the "Languages" intent by entering the query I know French. Then, answer the question How long have your known $language? with about 2 weeks.

![80dafbbf985df5a8.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/0315682fe974cdd71b8e73de58c93b6015352cf8d2b77a8f86f7504dc214615d.png)

Despite there being no response for the second query ("about 2 weeks"), we can see our query is matched to the correct intent ("Languages - custom") and the duration parameter is correctly parsed ("2 weeks").

### Intents and contexts

Now that your follow-up intent is being matched correctly, you need to add a response. In "Languages - custom" you've only asked for the duration the user has known the language, and not the referenced language itself.

To respond with a parameter gathered from the "Languages" intent, you need to know how follow-up intents work. Follow-up intents use contexts to keep track of if a parent intent has been triggered. If you inspect the "Languages" intent, you'll see "Languages-followup" listed as an **Output context**, prefaced by the number 2:

![773bf5df6bce12fe.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/293aba0568704e17d9e1769f20eb53c2b42db64cc8879705a778205ac4da8bfb.png)

After the "Languages" intent is matched, the context "Languages-followup" is attached to the conversation for two turns. Therefore, when the user responds to the question, "How long have you known $language?", the context "Languages-followup" is active. Any intents that have the same **Input context** are heavily favored when Dialogflow matches intents.

1. Click on **Intents** in the left navigation and then click on the "Languages - custom" intent.

![f72f77ab8f711d5.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/d2ce16b04c193955e3fbed5b78362fdc5bbfbf2677a44bbcb8ae65712cbc1621.png)

You can see that the intent has the same input context ("Languages-followup") as the output context of "Languages". Because of this, "Languages - custom" is much more likely to be matched after the "Languages" intent is matched.

### Contexts and parameters

Contexts store parameter values, which means you can access the values of parameters defined in the "Languages" intent in other intents like "Languages - custom".

1. Add the following response to the "Languages - custom" intent and click the **Save** button:

* I can't believe you've known #languages-followup.language for $duration!

![5641f3e48acdcbf8.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/200e2542c27b2e31127a5bc88a84057b8ec14a681f701eee15cf477774356c38.png)

**Save** the changes. Now you can query your agent again and get the proper response. First enter "I know French", and then respond to the question with "1 month".

![4d80e08cebfed7c5.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/dee854cbd5f92365457bab9bae4da82bb1a44f92fce4be3c699dc71cc6c7fdff.png)

You should see that the language parameter value is retrieved from the context.

### Next steps

If you have any questions or thoughts, let us know on the [Dialogflow Google Plus Community](https://plus.google.com/communities/103318168784860581977). We'd love to hear from you!

Now that you've completed your first agent, you can extend your response logic with fulfillment and consider which additional platforms you want to support via Dialogflow's one-click integrations.

Fulfillment allows you to provide programmatic logic behind your agent for gathering third-party data or accessing user-based information.

* [Fulfillment](https://dialogflow.com/docs/fulfillment)
* [How to get started with fulfillment](https://dialogflow.com/docs/how-tos/getting-started-fulfillment)
* [Integrate your service with fulfillment](https://dialogflow.com/docs/getting-started/integrate-services)
* [Integrate your service with Actions on Google](https://dialogflow.com/docs/getting-started/integrate-services-actions-on-google)

Dialogflow's integrations make your agent available on popular platforms like Facebook Messenger, Slack and Twitter.

* [Integrations Overview](https://dialogflow.com/docs/integrations/)
* [Facebook Messenger](https://dialogflow.com/docs/integrations/facebook)
* [Slack](https://dialogflow.com/docs/integrations/slack)
* [Twitter](https://dialogflow.com/docs/integrations/twitter)

You might also want to check out:

* [Contexts](https://dialogflow.com/docs/contexts)
* [Dialogflow and Actions on Google](https://dialogflow.com/docs/getting-started/dialogflow-actions-on-google)

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.





# Further materials

<https://cloud.google.com/natural-language/docs/classify-text-tutorial>



# [[eof]]



check:
<https://colab.research.google.com/github/amygdala/tensorflow-workshop/blob/master/workshop_sections/high_level_APIs/mnist_cnn_custom_estimator/cnn_mnist_tf.ipynb>



[[todo]]:

* make python and html files of all **notebooks** in all courses!
* make pdfs of all **.md-files i**n all courses! (add to .gitignore)
* make "final" model with all bits and pieces:
  * tensorboard infos that work
  * get model from cloud storage to local disk
  * load model from disk and make predictions
  * get "local" evaluation metric

