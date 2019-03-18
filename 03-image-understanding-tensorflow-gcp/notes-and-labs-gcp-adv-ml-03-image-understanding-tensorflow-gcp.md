# Image Understanding with Tensorflow on GCP



## to download materials

software to download materials: <https://github.com/coursera-dl/coursera-dl>

link to coursera material: <https://www.coursera.org/learn/end-to-end-ml-tensorflow-gcp/lecture/wWBkE/effective-ml>

```bash
coursera-dl -u "ingo.nader@gmail.com" image-understanding-tensorflow-gcp -sl en --download-quizzes --download-notebooks --about

#coursera-dl -u "ingo.nader@gmail.com" gcp-production-ml-systems -sl "en,de" --download-quizzes --download-notebooks --about
```

Done with modified version of coursera-dl: <https://github.com/coursera-dl/coursera-dl/issues/702>



## Notes

* Do labs in Google Chrome (recommended)
* Open Incognito window, go to <http://console.cloud.google.com/>



# Lab 1: [ML on GCP C8] Image Classification with a Linear Model

## Overview

*Duration is 1 min*

In this lab, you will define a simple linear image model on MNIST using the Estimator API to do image classification.

### **What you learn**

In this lab, you will learn how to:

* Import the training dataset of MNIST handwritten images
* Reshape and preprocess the image data
* Setup your linear classifier model with 10 classes (one for each possible digit 0 through 9)
* Define and create your EstimatorSpec in tensorflow to create your custom estimator
* Define and run your train_and_evaluate function to train against the input dataset of 60,000 images and evaluate your model's performance

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

## MNIST Image Classification using a linear model

*Duration is 15 min*

**Step 1**

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 08_image > labs** and open **mnist_linear.ipynb**.

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

**Step 2**

In Datalab, click on **Clear | Clear all Cells**. Now read the narrative and execute each cell in turn:

* Some lab tasks include starter code. In such cells, look for lines marked *#TODO*. Specifically, you need to write code to define the *linear_model* and the *eval_input_fn* function.
* If you need more help, you may take a look at the complete solution by navigating to : **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 08_image** and open the notebook: **mnist_linear.ipynb**

Note: When doing copy/paste of python code, please be careful about indentation

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.



# Lab 2: [ML on GCP C8] Image Classification with a Deep Neural Network Model

## Overview

*Duration is 1 min*

In this lab, you will define a deep neural network model to do image classification.

### What you learn

In this lab, you will learn how to:

* Import the training dataset of MNIST handwritten images
* Reshape and preprocess the image data
* Setup your neural network model with 10 classes (one for each possible digit 0 through 9)
* Define and create your EstimatorSpec in tensorflow to create your custom estimator
* Define and run your train_and_evaluate function to train against the input dataset of 60,000 images and evaluate your model's performance

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

## MNIST Image Classification

*Duration is 15 min*

This lab is organised a little different from lab 1 (linear model). The model code is packaged as a separate python module. You will first complete the model code and then switch to the notebook to set some parameters and run the training job.

**Step 1**

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 08_image > labs > mnistmodel > trainer** and open **model.py**.

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

**Step 2**

Scroll down to *dnn_model* where you have to replace the #TODO with code to define this dnn model.

If you need more help, you may take a look at the complete solution by navigating to : **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 08_image > mnistmodel > trainer** and open **model.py**

**Step 3**

Now that you have defined your dnn_model, you are ready to run the training job.

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 08_image > labs** and open **mnist_models.ipynb**.

**Step 4**

In Datalab, click on **Clear | Clear all Cells**.

**Step 5**

In the first cell, make sure to replace the project id, bucket and region with your qwiklabs project id, your bucket, and bucket region respectively. Also, change the MODEL_TYPE to *dnn*.

**Step 6**

Now read the narrative in the following cells and execute each cell in turn.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.



# Lab 3: [ML on GCP C8] Image Classification with a DNN Model with Dropout

## Overview

*Duration is 1 min*

In this lab, you will define a DNN with dropout on MNIST to do image classification.

### **What you learn**

In this lab, you will learn how to:

* Import the training dataset of MNIST handwritten images
* Reshape and preprocess the image data
* Setup your neural network model with 10 classes (one for each possible digit 0 through 9)
* Add a Dropout layer
* Define and create your EstimatorSpec in tensorflow to create your custom estimator
* Define and run your train_and_evaluate function to train against the input dataset of 60,000 images and evaluate your model's performance

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

## Create storage bucket and store data file

*Duration is 2 min*

Create a bucket using the GCP console:

**Step 1**

In your GCP Console, click on the **Navigation menu** (![menu.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/2f22c727923782d61d44dd691c367624adaf6ddd6c33a5b6be24f43b3c913d3b.png)), and select **Storage**.

**Step 2**

Click on **Create bucket**.

**Step 3**

Choose a Regional bucket and set a unique name (use your project ID because it is unique).

**Step 4**

For location, select from the following: `asia-northeast1, europe-west1, us-central1, us-east1`

**Step 5**

Click **Create**.

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

## MNIST Image Classification

*Duration is 15 min*

This lab uses the same files as lab 2 (dnn model). The model code is packaged as a separate python module. You will first complete the model code and then switch to the notebook to set some parameters and run the training job.

**Step 1**

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 08_image > labs > mnistmodel > trainer** and open **model.py**.

**Note:** If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

**Step 2**

Scroll down to *dnn_dropout_model* where you have to replace the *#TODO*with code to define this dnn model with dropout.

If you need more help, you may take a look at the complete solution by navigating to : **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 08_image > mnistmodel > trainer** and open **model.py**.

**Step 3**

Now that you have defined your dnn_dropout_model, you are ready to run the training job.

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 08_image > labs** and open **mnist_models.ipynb**.

**Step 4**

In Datalab, click on **Clear | Clear all Cells**.

**Step 5**

In the first cell, make sure to replace the project ID, bucket and region with your Qwiklabs project ID, your bucket, and bucket region respectively. Also, change the MODEL_TYPE to *dnn_dropout*.

**Step 6**

Now read the narrative in the following cells and execute each cell in turn.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.





# Lab 4: [ML on GCP C8] Image Classification with a CNN Model

## Overview

*Duration is 1 min*

In this lab, you will define a cnn model on MNIST to do image classification.

### **What you learn**

In this lab, you will learn how to:

* Import the training dataset of MNIST handwritten images
* Reshape and preprocess the image data
* Setup your CNN with 10 classes
* Create convolutional and pooling layers + softmax function
* Define and create your EstimatorSpec in tensorflow to create your custom estimator
* Define and run your train_and_evaluate

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

## MNIST Image Classification

*Duration is 15 min*

This lab uses the same files as labs 2 and 3 (dnn, dnn_dropout models). The model code is packaged as a separate python module. You will first complete the model code and then switch to the notebook to set some parameters and run the training job.

**Step 1**

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 08_image > labs > mnistmodel > trainer** and open **model.py**.

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

**Step 2**

Scroll down to *cnn_model* where you have to replace the *#TODO*s with code to define this cnn model.

If you need more help, you may take a look at the complete solution by navigating to : **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 08_image > mnistmodel > trainer** and open **model.py**

**Step 3**

Now that you have defined your *cnn_model*, you are ready to run the training job.

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 08_image > labs** and open **mnist_models.ipynb**.

**Step 4**

In Datalab, click on **Clear | Clear all Cells.**

**Step 5**

In the first cell, make sure to replace the project id, bucket and region with your qwiklabs project id, your bucket, and bucket region respectively. Also, change the MODEL_TYPE to *cnn*.

**Step 6**

Now read the narrative in the following cells and execute each cell in turn.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.



# Lab 5: Image Augmentation in TensorFlow

## 1. Overview

### What you need

You must have completed Lab 0 and have the following:

* Logged into GCP Console with your Qwiklabs generated account
* launched Datalab and cloned the training-data-analyst repo

### What you learn

In this lab, you do image classification from scratch on a flowers dataset using the Estimator API.

## 2. Flowers Image Classification

### Step 1

In Cloud Datalab, click on the Home icon, and then navigate to 

* notebooks/training-data-analyst/courses/machine_learning/deepdive/08_image/labs and open 
* `flowers_fromscratch.ipynb`.

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘datalab connect mydatalabvm' in your new Cloud Shell. Once connected, try the above step again.

### Step 2

In Datalab, click on Clear | All Cells. Now read the narrative and execute each cell in turn.



# Lab 6: [ML on GCP C8] Image Classification Transfer Learning with Inception v3

2 hours

Free





Rate Lab

## Overview

In this lab, you carry out a transfer learning example based on Inception-v3 image recognition neural network.

### **What you learn**

* How a transfer learning works
* How to use Cloud Dataflow for a batch processing of image data
* How to use Cloud ML to train a classification model
* How to use Cloud ML to provide a prediction API service

## Introduction

*Duration is 2 min*

Transfer learning is a machine learning method which utilizes a pre-trained neural network. For example, the image recognition model called[Inception-v3](https://arxiv.org/abs/1512.00567) consists of two parts:

* Feature extraction part with a convolutional neural network.
* Classification part with fully-connected and softmax layers.

The pre-trained Inception-v3 model achieves state-of-the-art accuracy for recognizing general objects with 1000 classes, like "Zebra", "Dalmatian", and "Dishwasher". The model extracts general features from input images in the first part and classifies them based on those features in the second part.

![bfea25ba557fbffc.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/e7e08f8bc7653d63a33d14cfc8afef2052c0e8323eed82c7b76c0cd2be17d4c8.png)

In transfer learning, when you build a new model to classify your original dataset, you reuse the feature extraction part and re-train the classification part with your dataset. Since you don't have to train the feature extraction part (which is the most complex part of the model), you can train the model with less computational resources and training time.

## Setup

*Duration is 5 min*

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

## Requirements

### Step 1: Enable the Cloud Machine Learning Engine API

Enable the Cloud Machine Learning Engine API.

* Navigate to <https://console.cloud.google.com/apis/library>.
* Search "Cloud Machine Learning Engine"
* Click on ENABLE button if necessary

### Step 2: Enable the Cloud Dataflow API

Enable the Cloud Dataflow API.

* Navigate to <https://console.cloud.google.com/apis/library>.
* Search "Dataflow API"
* Click on ENABLE button if necessary

### Step 3: Launch CloudShell

Now let's open the cloud shell. The cloud shell icon is at the top right:

![8206c366e1f66c6e.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/07096b2c16aac8c23a5fe1f0be0c402f43e2039e2b2ff8001a1b77093dffd49e.png)

A cloud shell session will open inside a new frame at the bottom of your browser.

![29c891701d874b22.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/3af82e23d9d597dc24d2972590fecff808216852c4943046a0584d1fb909ed54.png)

### Step 4: Install Cloud ML SDK

Install Cloud ML SDK, and modify the PATH.

```bash
sudo pip install google-cloud-dataflow
sudo pip install image
export PATH=${HOME}/.local/bin:${PATH}
```

#### Note:

* Some errors occur during installation:

```
[...]
  Building wheel for ply (setup.py) ... done
  Stored in directory: /root/.cache/pip/wheels/f2/21/c0/f0056cc96847933daa961a19eb59a2ecd0228fdbe3376e7a68
Successfully built google-cloud-dataflow dill pyyaml pyvcf hdfs avro httplib2 proto-google-cloud-pubsub-v1 googledatastore proto-google-cloud-datastore-v1 docopt gapic-google-cloud-pubsub-v1 ply
google-cloud-storage 1.14.0 has requirement google-cloud-core<0.30dev,>=0.29.0, but you'll have google-cloud-core 0.25.0 which is incompatible.
gapic-google-cloud-pubsub-v1 0.15.4 has requirement oauth2client<4.0dev,>=2.0.0, but you'll have oauth2client 4.1.3 which is incompatible.
proto-google-cloud-pubsub-v1 0.15.4 has requirement oauth2client<4.0dev,>=2.0.0, but you'll have oauth2client 4.1.3 which is incompatible.
google-cloud-translate 1.3.3 has requirement google-cloud-core<0.30dev,>=0.29.0, but you'll have google-cloud-core 0.25.0 which is incompatible.
google-cloud-logging 1.10.0 has requirement google-cloud-core<0.30dev,>=0.29.0, but you'll have google-cloud-core 0.25.0 which is incompatible.
google-gax 0.15.16 has requirement future<0.17dev,>=0.16.0, but you'll have future 0.17.1 which is incompatible.
proto-google-cloud-datastore-v1 0.90.4 has requirement oauth2client<4.0dev,>=2.0.0, but you'll have oauth2client 4.1.3 which is incompatible.
google-cloud-datastore 1.7.3 has requirement google-cloud-core<0.30dev,>=0.29.0, but you'll have google-cloud-core 0.25.0 which is incompatible.
google-cloud-spanner 1.7.1 has requirement google-cloud-core<0.30dev,>=0.29.0, but you'll have google-cloud-core 0.25.0 which is incompatible.
googledatastore 7.0.1 has requirement oauth2client<4.0.0,>=2.0.1, but you'll have oauth2client 4.1.3 which is incompatible.
Installing collected packages: dill, six, pyyaml, typing, pyvcf, docopt, hdfs, avro, httplib2, monotonic, fasteners, google-apitools, proto-google-cloud-pubsub-v1, google-cloud-core, google-cloud-bigquery, ply, google-gax, gapic-google-cloud-pubsub-v1, google-cloud-pubsub, proto-google-cloud-datastore-v1, googledatastore, apache-beam, google-cloud-dataflow
[...]
```





### Step 5: Download tutorial files

Download tutorial files and set your current directory.

```bash
git clone https://github.com/GoogleCloudPlatform/cloudml-samples
cd cloudml-samples/flowers
```

Using code from other repo:

```bash
#export GITREPO=gcp-ml-02-advanced-ml-with-tf-on-gcp
#cd /content/datalab/
#git clone https://github.com/ingonader/${GITREPO}.git
#cd $GITREPO
#git config user.email "ingo.nader@gmail.com"
#git config user.name "Ingo Nader"
#cd 03-image-understanding-tensorflow-gcp/
#cd flowersmodeltpu
## nope, doesn't work either
```



## Learn the Dataset

*Duration is 2 min*

![7221191ec60f55f6.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/6bdb33f4cada22b9d59015e98c3d95184b87ca2bd394d66080292f724b50e9c8.png)

[Sunflowers](https://www.flickr.com/photos/calliope/1008566138/) *by Liz West is licensed under* [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)

You have about 3,600 flower images with five categories. The five category labels are listed in the dictionary file ( [gs://cloud-ml-data/img/flower_photos/dict.txt](https://storage.cloud.google.com/cloud-ml-data/img/flower_photos/dict.txt)) as below:

```bash
daisy
dandelion
roses
sunflowers
tulips
```

This file is used to translate labels into internal ID numbers in the following processes such as daisy=0, dandelion=1, etc.

The images are randomly split into a training set with 90% data and an evaluation set with 10% data. Each of them are listed in CSV files:

* Training set: [train_set.csv](https://storage.cloud.google.com/cloud-ml-data/img/flower_photos/train_set.csv)
* Evaluation set: [eval_set.csv](https://storage.cloud.google.com/cloud-ml-data/img/flower_photos/eval_set.csv)

The CSV files have the following format consisting of image URIs and labels:

```bash
gs://cloud-ml-data/img/flower_photos/dandelion/17388674711_6dca8a2e8b_n.jpg,dandelion
gs://cloud-ml-data/img/flower_photos/sunflowers/9555824387_32b151e9b0_m.jpg,sunflowers
gs://cloud-ml-data/img/flower_photos/daisy/14523675369_97c31d0b5b.jpg,daisy
gs://cloud-ml-data/img/flower_photos/roses/512578026_f6e6f2ad26.jpg,roses
gs://cloud-ml-data/img/flower_photos/tulips/497305666_b5d4348826_n.jpg,tulips
...
```

You input these images into the feature extraction part of Inception-v3 which converts the image data into feature vectors consisting of 2048 float values for each image. A feature vector represents the features of the image in an abstract manner. You can better classify images based on these vector values rather than raw image data.

## Preprocess Images with Cloud Dataflow

*Duration is 54 min*

### Preprocess the Evaluation Set

You use Cloud Dataflow to automate the feature extraction process. First, you do it for the evaluation dataset.

#### Step 1

Set variables to specify the output path and the dictionary file on Cloud Storage.

```bash
DICT_FILE=gs://cloud-ml-data/img/flower_photos/dict.txt
PROJECT=$(gcloud config list project --format "value(core.project)")
BUCKET="gs://${PROJECT}-flower"
GCS_PATH="${BUCKET}/${USER}"
```

#### Step 2

Create a bucket for the output files and submit a Cloud Dataflow job to process the evaluation set.

```bash
gsutil mb $BUCKET
python trainer/preprocess.py \
  --input_dict "$DICT_FILE" \
  --input_path "gs://cloud-ml-data/img/flower_photos/eval_set.csv" \
  --output_path "${GCS_PATH}/preproc/eval" \
  --cloud \
  --num_workers 5
```

Doesn't work:

```
/usr/local/lib/python2.7/dist-packages/tensorflow/contrib/lite/python/__init__.py:26: PendingDeprecationWarning: WARNING: TF Lite has moved from tf.contrib.lite to tf.lite. Please update your imports. This will be a breaking error in TensorFlow version 2.0.
  _warnings.warn(WARNING, PendingDeprecationWarning)
/usr/local/lib/python2.7/dist-packages/apache_beam/io/gcp/gcsio.py:176: DeprecationWarning: object() takes no parameters
  super(GcsIO, cls).__new__(cls, storage_client))
WARNING:root:Retry with exponential backoff: waiting for 4.49787994343 seconds before retrying exists because we caught exception: SSLHandshakeError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:661)
 Traceback for above exception (most recent call last):
  File "/usr/local/lib/python2.7/dist-packages/apache_beam/utils/retry.py", line 180, in wrapper
    return fun(*args, **kwargs)
  File "/usr/local/lib/python2.7/dist-packages/apache_beam/io/gcp/gcsio.py", line 374, in exists
    self.client.objects.Get(request)  # metadata
  File "/usr/local/lib/python2.7/dist-packages/apache_beam/io/gcp/internal/clients/storage/storage_v1_client.py", line 951, in Get
    download=download)
  File "/usr/local/lib/python2.7/dist-packages/apitools/base/py/base_api.py", line 720, in _RunMethod
    http, http_request, **opts)
  File "/usr/local/lib/python2.7/dist-packages/apitools/base/py/http_wrapper.py", line 356, in MakeRequest
    max_retry_wait, total_wait_sec))
  File "/usr/local/lib/python2.7/dist-packages/apitools/base/py/http_wrapper.py", line 304, in HandleExceptionsAndRebuildHttpConnections
    raise retry_args.exc
```





By clicking on the Cloud **Dataflow** menu on the Cloud Console, you find the running job and the link navigates to the data flow graph as below:

![3a348c6bd13df85b.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/a1d3f8f5c3c729f14a5191ae60d753b71f14a79bc5783d00dc152b4186c83eab.png)

**Note**: The job takes around 5-30 minutes (depending on the environment) to finish. You can proceed with the next subsection to process the training set in parallel.

### Preprocess the Training Set

Since the job takes some time to finish, you open a new tab in the cloud shell and submit another job to process the training set in parallel.

#### Step 3

Set your current directory in the new tab.

```bash
cd $HOME/cloudml-samples/flowers
```

#### Step 4

Set variables to specify the output path and the dictionary file on Cloud Storage.

```bash
DICT_FILE=gs://cloud-ml-data/img/flower_photos/dict.txt
PROJECT=$(gcloud config list project --format "value(core.project)")
BUCKET="gs://${PROJECT}-flower"
GCS_PATH="${BUCKET}/${USER}"
```

#### Step 5

Submit a Cloud Dataflow job to process the training set.

```bash
python trainer/preprocess.py \
  --input_dict "$DICT_FILE" \
  --input_path "gs://cloud-ml-data/img/flower_photos/train_set.csv" \
  --output_path "${GCS_PATH}/preproc/train" \
  --cloud
```

**Note**: The whole process takes around 10 to 45 minutes (depending on the environment) to finish. Please wait for both jobs to finish successfully.

### Confirm the Preprocessed Files

When both jobs finished, the preprocessed files are stored in the following storage paths.

#### Step 6

Confirm that preprocessed files for the evaluation set are created.

```bash
gsutil ls "${GCS_PATH}/preproc/eval*"
```

You see the following files are stored in Cloud Storage.

```bash
gs://<Your project ID>-flower/enakai/preproc/eval-00000-of-00043.tfrecord.gz
gs://<Your project ID>-flower/enakai/preproc/eval-00001-of-00043.tfrecord.gz
gs://<Your project ID>-flower/enakai/preproc/eval-00002-of-00043.tfrecord.gz
...
```

#### Step 7

Confirm that preprocessed files for the training set are created.

```bash
gsutil ls "${GCS_PATH}/preproc/train*"
```

You see the following files are stored in Cloud Storage.

```bash
gs://<Your project ID>-flower/enakai/preproc/train-00000-of-00062.tfrecord.gz
gs://<Your project ID>-flower/enakai/preproc/train-00001-of-00062.tfrecord.gz
gs://<Your project ID>-flower/enakai/preproc/train-00002-of-00062.tfrecord.gz
...
```

## Train the Model with Cloud ML

*Duration is 15 min*

The next step is to train the classification part of the model using the preprocessed data. The following diagram shows the relationship between the preprocessing and the training.

![4a077a942db85cb3.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/4431cd0bb49a7944e1e763495a534b13bfb19a4fd2c1f54c17d01013604057b4.png)

#### Step 1

Submit a Cloud ML job to train the classification part of the model:

```bash
JOB_ID="flowers_${USER}_$(date +%Y%m%d_%H%M%S)"
gcloud ml-engine jobs submit training "$JOB_ID" \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket "$BUCKET" \
  --region us-central1 \
  -- \
  --output_path "${GCS_PATH}/training" \
  --eval_data_paths "${GCS_PATH}/preproc/eval*" \
  --train_data_paths "${GCS_PATH}/preproc/train*" \
  --label_count 5
```

**Note**: `JOB_ID` can be arbitrary but you cannot reuse the same one.

#### Step 2

Click on the **Navigation menu** > **ML Engine** on the Cloud Console, find the running job and navigate to the **Logs** > **View Logs** to see log messages. The training process takes about 10 minutes. After 1000 training steps, you see the messages as below:

```bash
INFO    2016-12-28 14:19:19 +0900       master-replica-0                Eval, step 1000:
INFO    2016-12-28 14:19:19 +0900       master-replica-0                - on train set 0.005, 1.000
INFO    2016-12-28 14:19:19 +0900       master-replica-0                -- on eval set 0.248, 0.937
INFO    2016-12-28 14:19:20 +0900       master-replica-0                Exporting prediction graph to gs://<Your project ID>-flower/<USER>/training/model
```

This means the model achieved 100% accuracy for the training set and 93.7% accuracy for the evaluation set. (The final accuracy may vary in each training.) The last message indicates that the whole model including the feature extraction part is exported so that you can use the model to classify new images without preprocessing them.

## Deploy the Trained Model for Predictions

*Duration is 10 min*

### Deploy the Trained Model

You deploy the exported model to provide a prediction API.

#### Step 1

Create a prediction API with the specified model name.

```bash
MODEL_NAME=flowers
gcloud ml-engine models create ${MODEL_NAME} --regions=us-central1
```

**Note**: `MODEL_NAME` can be arbitrary but you cannot reuse the same one.

#### Step 2

Create and set a default version of the model with the specified version name.

```bash
VERSION_NAME=v1
gcloud ml-engine versions create \
  --origin ${GCS_PATH}/training/model/ \
  --model ${MODEL_NAME} \
  ${VERSION_NAME}
gcloud ml-engine versions set-default --model ${MODEL_NAME} ${VERSION_NAME}
```

**Note**: `VERSION_NAME` can be arbitrary but you cannot reuse the same one. The last command is not necessary for the first version since it automatically becomes the default. It's here as a good practice to set the default explicitly.

**Important**: It may take a few minutes for the deployed model to become ready. Until it becomes ready, it returns 503 error.

### Create a JSON Request File

Now you can send an image data to get a classification result. First, you need to convert raw image files into a JSON request file.

#### Step 3

Download two sample images.

```bash
gsutil cp gs://cloud-ml-data/img/flower_photos/tulips/4520577328_a94c11e806_n.jpg flower1.jpg
gsutil cp gs://cloud-ml-data/img/flower_photos/roses/4504731519_9a260b6607_n.jpg flower2.jpg
```

#### Step 4

Convert the raw images to a single JSON request file.

```bash
python -c 'import base64, sys, json; \
  img = base64.b64encode(open(sys.argv[1], "rb").read()); \
  print json.dumps({"key":"1", "image_bytes": {"b64": img}})' \
  flower1.jpg > request.json
python -c 'import base64, sys, json; \
  img = base64.b64encode(open(sys.argv[1], "rb").read()); \
  print json.dumps({"key":"2", "image_bytes": {"b64": img}})' \
  flower2.jpg >> request.json
```

In this example, the following images are encoded in base64 and stored in a dictionary associated with key values 1 and 2 respectively.

![ac9eb40193038e37.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/a3a03e910f97ad8569bdaf4480b9971c3451d4a276a90a9b4ab51902e17e0bea.jpg) ![cfcce589272dcfd2.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/ccecdbf6f71eac9e492745dbf827c0f1fa84b347bca132bb5e41b80c865e9a50.jpg)

### Send a Request to the Prediction API

#### Step 5

Use the gcloud command to send a request.

```bash
gcloud ml-engine predict --model ${MODEL_NAME} --json-instances request.json
```

The command returns the prediction result as below:

```bash
KEY  PREDICTION  SCORES
1    4           [1.5868581115796587e-08, 5.666522540082042e-08, 3.1425850011146395e-06, 3.0022348496139273e-10, 0.9999967813491821, 8.086380454130904e-09]
2    3           [0.00016239925753325224, 0.007603001315146685, 0.2902684807777405, 0.6873179078102112, 0.01457294262945652, 7.533563621109352e-05]
```

The returned message shows the key value, prediction and scores. You can use the key value to associate the result with the input image. The prediction is shown in the internal ID as explained in Learn the Dataset section. 4 and 3 correspond to tulips and sunflowers respectively. So the prediction for the second image is incorrect in this example.

You can see more details in scores which show the estimated probabilities for each category. For the first image, the score for ID 4 is almost 1.0. On the other hand, for the second image, the score for ID 3 (sunflowers) is about 0.69 whereas the score for ID 2 (roses) is about 0.29. Compared to the first image, you can see that the prediction for the second image is more uncertain.

## Conclusion

*Duration is 1 min*

### What we've covered

* How a transfer learning works
* How to use Cloud Dataflow for a batch processing of image data
* How to use Cloud ML to train a classification model
* How to use Cloud ML to provide a prediction API service

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.



# Lab 7 (optional): ML on GCP - TPU ResNet Lab

1 hour 30 minutes

Free





Rate Lab

## Overview

*Duration is 1 min*

In this lab, you will define a cnn model on MNIST to do image classification

### **What you learn**

In this lab, you will learn how to:

* Import the training dataset of MNIST handwritten images
* Reshape and preprocess the image data
* Setup your CNN with 10 classes
* Create convolutional and pooling layers + softmax function
* Define and create your EstimatorSpec in tensorflow to create your custom estimator
* Define and run your train_and_evaluate

## Setup

For each lab, you get a new GCP project and set of resources for a fixed time at no cost.

1. Make sure you signed into Qwiklabs using an **incognito window**.
2. Note the lab's access time (for example, ![img](https://cdn.qwiklabs.com/EMsbUpB%2FSsSGtqXC3aqAMFd4lxUhhGJdGw6%2B64K8Dh4%3D)) and make sure you can finish in that time block.

There is no pause feature. You can restart if needed, but you have to start at the beginning.

1. When ready, click ![img](https://cdn.qwiklabs.com/ltEm10Z7LAMKEbql8k0cPTtkbIC7qDzniO26taMtmCw%3D) **.**
2. Note your lab credentials. You will use them to sign in to Cloud Platform Console.![img](https://cdn.qwiklabs.com/SWhvFcEH3cqGWabnlhVPPJsj4BOPZUIRWCy8eUOtYfM%3D)
3. Click **Open Google Console**.
4. Click **Use another account** and copy/paste credentials for **this** lab into the prompts.

If you use other credentials, you'll get errors or **incur charges**

1. Accept the terms and skip the recovery resource page.

Do not click **End** unless you are finished with the lab or want to restart it. This clears your work and removes the project.

## Create storage bucket

*Duration is 2 min*

Create a bucket using the GCP console:

### **Step 1**

In your GCP Console, click on the Navigation menu (![img](https://cdn.qwiklabs.com/LyLHJ5I3gtYdRN1pHDZ2JK2vbd1sM6W2viT0OzyRPTs%3D)), and select **Storage**

### **Step 2**

Click on **Create Bucket**

### **Step 3**

Choose a Regional bucket and set a unique name (use your project id because it is unique). Then, click **Create**.

## Launch Cloud Datalab

To launch Cloud Datalab:

### **Step 1**

Open Cloud Shell. In the upper-right, click **Activate Cloud Shell** (![img](https://cdn.qwiklabs.com/sqKx45X8b2P7ygEtesyerKaHyXQGXOYNqXOqo%2Bl8nDA%3D)). And then click **Start Cloud Shell**.

### **Step 2**

In Cloud Shell, type:

```
gcloud compute zones list
```

Note: Please pick a zone in a geographically close region from the following: **us-east1, us-central1, asia-east1, europe-west1**. These are the regions that currently support Cloud ML Engine jobs. Please verify [here](https://cloud.google.com/ml-engine/docs/environment-overview#cloud_compute_regions) since this list may have changed after this lab was last updated. For example, if you are in the US, you may choose **us-east1-c** as your zone.

### **Step 3**

In Cloud Shell, type:

```
datalab create mydatalabvm --zone <ZONE>
```

Replace <ZONE> with a zone name you picked from the previous step.

Note: follow the prompts during this process.

Datalab will take about 5 minutes to start.

### **Step 4**

Look back at Cloud Shell, and follow any prompts. If asked for a ssh passphrase, just hit return (for no passphrase).

### **Step 5**

If necessary, wait for Datalab to finish launching. Datalab is ready when you see a message prompting you to do a "Web Preview".

### **Step 6**

Click on the **Web Preview** icon on the top-right corner of the Cloud Shell ribbon. After clicking on "Web preview", click on "Change port" and change the port number to 8081. Click "Change and Preview" to open the Datalab web interface.

![img](https://cdn.qwiklabs.com/YzKWe9v%2BrRQhMjdSi05hLwBpHpltc%2BAf4P7AvTCoJH8%3D)

![img](https://cdn.qwiklabs.com/CDt1IjvuSBjDt8hiTatVKmhz3ipa5Y8HepCRmUCsE0o%3D)

![img](https://cdn.qwiklabs.com/1CeV15GkPYccWcitjsyyKQ6TIJ2xngacAmcmVnYeu%2Bg%3D)

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell.

## Clone course repo within your Datalab instance

To clone the course repo in your Datalab instance:

### **Step 1**

In Cloud Datalab home page (browser), navigate into "**notebooks**" and add a new notebook using the icon ![img](https://cdn.qwiklabs.com/%2FvDMjDahhWqkynNCPyulnd5jUmdDfBJTwmjzZt%2FhmJk%3D) on the top left.

### **Step 2**

Rename this notebook as ‘**repocheckout**'.

### **Step 3**

In the new notebook, enter the following commands in the cell, and click on **Run** (on the top navigation bar) to run the commands:

```
%bash
git clone https://github.com/GoogleCloudPlatform/training-data-analyst
rm -rf training-data-analyst/.git
```

![img](https://cdn.qwiklabs.com/i443mZtLK9qMWGLmaG9dvKMtxS%2BqlT8ZX%2FlLE7xCNXs%3D)

### **Step 4**

Confirm that you have cloned the repo by going back to Datalab browser, and ensure you see the **training-data-analyst** directory. All the files for all labs throughout this course are available in this directory.

![img](https://cdn.qwiklabs.com/VZ1qcxgmJYrbYrV%2BrQpX1uY%2FR7pmd3lzbEp3kWKvqpU%3D)

## MNIST Image Classification

*Duration is 15 min*

This lab uses the same files as labs 2 and 3 (dnn, dnn_dropout models). The model code is packaged as a separate python module. You will first complete the model code and then switch to the notebook to set some parameters and run the training job.

### **Step 1**

In Cloud Datalab, click on the Home icon, and then navigate to

**training-data-analyst/quests/tpu/** and open [**flowers_resnet.ipynb**](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/quests/tpu/flowers_resnet.ipynb)

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

### **Step 2**

Now read the narrative in the following cells and execute each cell in turn.

# Lab [[?]] Training a ResNet Image Classifier from Scratch with TPUs on Cloud Machine Learning Engine

1 hour 30 minutes

Free





Rate Lab

## Overview:

In this lab you will train a state-of-the-art image classification model on your own data using Google's Cloud TPUs from CloudShell.

### What you learn

In this lab, you learn how to:

* Convert JPEG files into TensorFlow records
* Train a ResNet image classifier
* Deploy the trained model as a web service
* Invoke the web service by sending it a JPEG image

## A brief primer on ResNet and TPUs

The more layers you have in a neural network, the more accurate it should be at image classification. However, deep neural networks are harder to train -- in practice, this difficulty overwhelms the optimization algorithm and so, as you increase the number of layers, the training error starts to increase. One way to address this optimization problem is to introduce a "shortcut" connection that does an identity mapping and ask the optimizer to focus on the residual (or difference):

![b379c97ff0377d93.png](https://cdn.qwiklabs.com/icoelyuzhs4HBvgnnsxTC49ZdjBrYneZrqDpQSDV5Hg%3D)

*Image from:* [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

A neural network consisting of such blocks turns out to be easier to train even if it has just as many layers as a deep neural network without the shortcut connections. Such deep residual networks swept the field of image and object recognition competitions and are now considered state-of-the-art models for image analysis tasks.

Tensor Processing Units (TPUs) are application-specific integrated circuit (ASIC) hardware accelerators for machine learning. They were custom-designed by Google to carry out TensorFlow operations and can provide significant speedups in machine learning tasks that are compute-bound (rather than I/O bound). Training deep residual networks for image classification is one such task. In [independent tests conducted by Stanford University](https://dawn.cs.stanford.edu/benchmark/), the ResNet-50 model trained on a TPU was the fastest to achieve a desired accuracy on a standard datasets[1]. If you use TPUs on serverless infrastructure as Cloud ML Engine, this also translates to lower cost, since you pay only for what you use and don't have to keep any machines up and running.

![2a62e08cede0f5b6.png](https://cdn.qwiklabs.com/GwdWNM7HdxOXAjJeW%2F45Fv1NtY1IR0i06XQVTk%2B6R30%3D)

*TPUs can speed up training of state-of-the-art models.*

## Test your understanding





TPUs are custom designed to carry out what operations efficiently?





Stackdriver

*check*Tensorflow





Cloud Function





BigQuery

Submit







It's a good idea to use TPUs on machine learning tasks that are I/O bound.True False







TPUs provide the fastest, most cost-effective way to train state-of-the-art image models.True False



## Setup your environment

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

![Open Google Console](https://cdn.qwiklabs.com/Xq7gN%2BbO2%2FSfanAquKnvggu45xczKUb%2FdqCDGmOWqvw%3D)

If your lab provides other resource identifiers or connection-related information, it will appear on this panel as well.

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

## Using the Cloud Machine Learning Engine and Dataflow APIs

The Cloud Machine Learning Engine and Dataflow APIs have been enabled for you for this lab. To enable them in your own environmnet, follow these instructions.

Navigate to <https://console.cloud.google.com/apis/library>, search for **Cloud Machine Learning Engine** and click on ENABLE button if it is not already enabled.

Navigate to <https://console.cloud.google.com/apis/library>, search for **Dataflow API** and click on ENABLE button if it is not already enabled.

## Clone repository

In CloudShell, run the following to clone the repository:

```
git clone https://github.com/GoogleCloudPlatform/training-data-analyst
```

## Explore data

![7221191ec60f55f6.png](https://cdn.qwiklabs.com/a9sz9MraIrnVkBXpjD2VGEuHyivTlNZggCkvcktQ6cg%3D)

*Sunflowers* by Liz West is licensed under [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)

You have about 3,600 flower images in five categories. The images are randomly split into a training set with 90% data and an evaluation set with 10% data listed in CSV files:

* Training set: [train_set.csv](https://storage.cloud.google.com/cloud-ml-data/img/flower_photos/train_set.csv)
* Evaluation set: [eval_set.csv](https://storage.cloud.google.com/cloud-ml-data/img/flower_photos/eval_set.csv)

Explore the format and contents of the `train.csv` by running:

```
gsutil cat gs://cloud-ml-data/img/flower_photos/train_set.csv | head -5 > /tmp/input.csv
cat /tmp/input.csv
```

How would you find all the types of flowers in the training dataset?

Answer: The first field is the GCS location of the image file, and the second field is the label. In this case, the label is the type of flower.

The set of all the labels is called the dictionary. Run the following to find them:

```
gsutil cat gs://cloud-ml-data/img/flower_photos/train_set.csv  | sed 's/,/ /g' | awk '{print $2}' | sort | uniq > /tmp/labels.txt
cat /tmp/labels.txt
```

What does label=3 correspond to?

Answer: The label file is used to translate labels into internal ID numbers in the following processes such as daisy=0, dandelion=1, etc. (it is 0-based). So, label=3 would correspond to sunflowers. The code extracts the second field out of the CSV file and determines the unique list.

Because this Google Cloud Storage bucket is public, you can view the images using http. An GCS URI such as:

gs://cloud-ml-data/img/flower_photos/daisy/754296579_30a9ae018c_n.jpg

is published on http as: <https://storage.cloud.google.com/cloud-ml-data/img/flower_photos/daisy/754296579_30a9ae018c_n.jpg>

Click on the above link to view the image. What type of flower is it?

Answer: This is the first line of the train_set.csv, and the label states that it is a daisy. I hope that is correct!

## Convert JPEG images to TensorFlow records

We do not want our training to be limited by I/O speed, so let's convert the JPEG image data into TensorFlow records. These are an efficient format particularly suitable for batch reads by the machine learning framework.

The conversion will be carried in Cloud Dataflow, the serverless ETL service on Google Cloud Platform.

Get ResNet code by running:

```
cd ~/training-data-analyst/quests/tpu
bash ./copy_resnet_files.sh 1.9
```

**Note:** The 1.9 refers to the version of TensorFlow.

Examine what has been copied over by running:

```
ls mymodel/trainer
```

Notice that ResNet model code has been copied over from<https://github.com/tensorflow/tpu/tree/master/models/official/resnet>

There are a number of other pre-built models for various other tasks in that repository.

Now create an output bucket to hold the TensorFlow records.

From the GCP navigation menu, go to **Storage** > **Browser** and create a new bucket. The bucket name has to be [universally unique](https://cloud.google.com/storage/docs/naming).

Set BUCKET and PROJECT environment variables:

```
export BUCKET=<BUCKET>
export PROJECT=$(gcloud config get-value project)
echo $BUCKET $PROJECT
```

Install the Apache Beam Python package:

```
sudo pip install 'apache-beam[gcp]'
```

Apache Beam is the open-source library for code that is executed by Cloud Dataflow.

Now run the conversion program:

```
export PYTHONPATH=${PYTHONPATH}:${PWD}/mymodel
gsutil -m rm -rf gs://${BUCKET}/tpu/resnet/data
python -m trainer.preprocess \
       --train_csv gs://cloud-ml-data/img/flower_photos/train_set.csv \
       --validation_csv gs://cloud-ml-data/img/flower_photos/eval_set.csv \
       --labels_file /tmp/labels.txt \
       --project_id $PROJECT \
       --output_dir gs://${BUCKET}/tpu/resnet/data
```

Wait for the Dataflow job to finish. Navigate to **Navigation menu** > **Dataflow** and look at the submitted jobs. Wait for the recently submitted job to finish. This will take 15-20 minutes.

## Train model

Verify that the TensorFlow records exist by running:

```
gsutil ls gs://${BUCKET}/tpu/resnet/data
```

You should see `train-*` and `validation-*` files. If no files are present, wait for the Dataflow job in the previous section to finish.

Enable the Cloud TPU account:

```
bash enable_tpu_mlengine.sh
```

Submit the training job:

```
TOPDIR=gs://${BUCKET}/tpu/resnet
REGION=us-central1
OUTDIR=${TOPDIR}/trained
JOBNAME=imgclass_$(date -u +%y%m%d_%H%M%S)
echo $OUTDIR $REGION $JOBNAME
gsutil -m rm -rf $OUTDIR  # Comment out this line to continue training from the last time
gcloud ml-engine jobs submit training $JOBNAME \
 --region=$REGION \
 --module-name=trainer.resnet_main \
 --package-path=$(pwd)/mymodel/trainer \
 --job-dir=$OUTDIR \
 --staging-bucket=gs://$BUCKET \
 --scale-tier=BASIC_TPU \
 --runtime-version=1.9 \
 -- \
 --data_dir=${TOPDIR}/data \
 --model_dir=${OUTDIR} \
 --resnet_depth=18 \
 --train_batch_size=128 --eval_batch_size=32 --skip_host_call=True \
 --steps_per_eval=250 --train_steps=1000 \
 --num_train_images=3300  --num_eval_images=370  --num_label_classes=5 \
 --export_dir=${OUTDIR}/export
```

Wait for the ML Engine job to finish.

Navigate to <https://console.cloud.google.com/mlengine> and look at the submitted jobs. Wait for the recently submitted job to finish. *This will take 15-20 minutes.*

## Examine training outputs

View training graph by running the following to launch TensorBoard:

```
tensorboard --logdir gs://${BUCKET}/tpu/resnet/trained --port=8080
```

Now you can open TensorBoard.

From the **Web Preview** menu at the top of CloudShell, select **Preview on Port 8080**.

![aab2c519c5686f61.png](https://cdn.qwiklabs.com/Di9hIYZHfAvsgvdFyESNQ5rf0TKKSYhj%2BltPNtaPF%2Bs%3D)

As your models get larger, and you export more checkpoints, you may need to wait 1-2 minutes for TensorBoard to load the data.

**View training graph**

Change to scalar graphs and view the loss and top_1_accuracy plots.

* Does the loss curve show that the train loss has plateaued?
* Does the evaluation loss indicate overfitting?
* Is the top_1_accuracy sufficient?
* How would you use the answers to the above questions?

Answers:

If the loss curve has not plateaued, re-run the training job for more training steps. Make sure you are not deleting the output directory, so that the training commences from the previous point.

![45a6dfedd82cf324.png](https://cdn.qwiklabs.com/IhUhh2D0y%2BGSy3lYJ%2BAF2zGQF49eY2qO%2B9241pEvkKM%3D)

If the evaluation loss (blue curve) is much higher than the training loss (orange curve), especially if the evaluation loss starts to increase, stop training (do an early-stop), reduce the ResNet model depth, or increase the size of your datasets.

If the top_1_accuracy is insufficient, increase the size of your dataset.

![dfb2993675727a79.png](https://cdn.qwiklabs.com/VNtI%2FhEwvuUquQHCHC%2BjihjPdfWrPZ0HvdvESG7L1SI%3D)

## Deploy model

**View exported model**

In CloudShell, press **Ctrl**+**C** if necessary to get back to the command prompt and run:

```
gsutil ls gs://${BUCKET}/tpu/resnet/trained/export/
```

Deploy trained model as a web service by running:

```
MODEL_NAME="flowers"
MODEL_VERSION=resnet
MODEL_LOCATION=$(gsutil ls gs://${BUCKET}/tpu/resnet/trained/export/ | tail -1)
echo "Deleting and deploying $MODEL_NAME $MODEL_VERSION from $MODEL_LOCATION ... this will take a few minutes"
gcloud ml-engine models create ${MODEL_NAME} --regions $REGION
gcloud ml-engine versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version=1.9
```

*This will take 4-5 minutes.*

Invoke the model:

```
python invoke_model.py  --project=$PROJECT --jpeg=gs://cloud-ml-data/img/flower_photos/sunflowers/1022552002_2b93faf9e7_n.jpg
```

The first image will take about a minute as service loads. Subsequent calls will be faster. Try other images.

## Learn More / Next Steps

Please see [this blog post](https://cloud.google.com/blog/big-data/2018/07/how-to-train-a-resnet-image-classifier-from-scratch-on-tpus-on-cloud-ml-engine) for more details about ResNet image classifier.

## Clean up

When you take a lab, all of your resources will be deleted for you when you're finished. But in the real world, you need to do it yourself to avoid incurring charges.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.





# Lab 8: [ML on GCP C8] Training with Pre-built ML Models using Cloud Vision API and AutoML

2 hours

Free





Rate Lab

## Overview

*Duration is 1 min*

In this lab, you will experiment with pre-built models so there's no coding. First we'll start with the pre-trained Vision API where we don't need to bring our own data and then we'll progress into AutoML for more sophisticated custom labelling that we need.

### **What you learn**

In this lab, you learn how to:

* Setup API key for ML Vision API
* Invoke the pretrained ML Vision API to classify images
* Review label predictions from Vision API
* Train and evaluate custom AutoML Vision image classification model
* Predict with AutoML on new image

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

## Enable Vision API and create API Key

*Duration is 1 min*

To get an API key:

**Step 1**

In your GCP Console, click on the **Navigation menu** (![menu.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/2f22c727923782d61d44dd691c367624adaf6ddd6c33a5b6be24f43b3c913d3b.png)), select **APIs and services** and select **Library**.

**Step 2**

In the search box, type **vision** to find the **Cloud Vision API** and click on the hyperlink.

![7c01378ef4631c52.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/3a5a849c18cd538c64c74d29c476ad6e9a9f4d577cc32d2b54827e4cc0e3681b.png)

**Step 3**

Click **Enable** if necessary.

![54d492d83efe122b.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/5214738934a89038e90399a8ec20ced4988c0cbe6de2fc4fd0fede0266ee0444.png)

**Step 4**

In your GCP Console, click on the **Navigation menu** (![menu.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/2f22c727923782d61d44dd691c367624adaf6ddd6c33a5b6be24f43b3c913d3b.png)), select **APIs & Services** and select **Credentials**.

**Step 5**

If you do not already have an API key, click the **Create credentials** button and select **API key**. Once created, copy the API key and then click **Close**.

![bc4940935c1bef7f.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/3b213bcba8eca9af81ea86f003e5580a51823d7cba111d41284d1fddc2e5c7bb.png)

**Step 6**

In Cloud Shell, export your API key as environment variable. Be sure to replace <YOUR_API_KEY> with the key you just copied.

```bash
export API_KEY=<YOUR_API_KEY>
```

## Create storage bucket and store data file

*Duration is 2 min*

Create a bucket using the GCP console:

**Step 1**

In your GCP Console, click on the **Navigation menu** (![menu.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/2f22c727923782d61d44dd691c367624adaf6ddd6c33a5b6be24f43b3c913d3b.png)), and select **Storage**.

**Step 2**

Click on **Create bucket**.

**Step 3**

Choose a Regional bucket and set a unique name (use your project ID because it is unique). Then, click **Create**.

**Step 4**

Download the image below by right-clicking and saving it locally (save it as cirrus.png):

![aa9c98d75e404b18.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/d04ea1f75190494df5156545caa101c2bf036122b607849b76798dbff93436d7.jpg)

**Step 5**

Upload the file you just downloaded into the storage bucket you just created using the **upload files** button.

**Step 6**

In Cloud Shell, run the command below to make the file publicly accessible.

```bash
export BUCKET=qwiklabs-gcp-3fcd3c73127d895b
gsutil acl ch -u AllUsers:R gs://$BUCKET/*

```

Click the **Public** link to confirm the file loads correctly (refresh bucket if needed).

![c5780f58f370ad37.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/9a21bbef2270f60a8adeabcefea83fd1f7ac28169f61a8b1d67615fab81a6f02.png)

## Label detection with Vision API

**Step 1**

First, you will create a Vision API request in a json file. Using gcloud or your preferred command line editors (nano, vim, or emacs) create a **request.json** file and inserting the following:

**Note:** Replace **my-bucket-name** with the name of your storage bucket.

```json
{
  "requests": [
      {
        "image": {
          "source": {
              "gcsImageUri": "gs://my-bucket-name/cirrus.png"
          }
        },
        "features": [
          {
            "type": "LABEL_DETECTION",
            "maxResults": 10
          }
        ]
      }
  ]
}
```

**Save** the file.

**Step 2**

The label detection method will return a list of labels (words) of what's in your image. Call the Vision API with curl:

```bash
curl -s -X POST -H "Content-Type: application/json" --data-binary @request.json  https://vision.googleapis.com/v1/images:annotate?key=${API_KEY}
```

Your response should look something like the following:

```json
{
        "responses": [{
                "labelAnnotations": [{
                        "mid": "/m/01bqvp",
                        "description": "sky",
                        "score": 0.9867201,
                        "topicality": 0.9867201
                }, {
                        "mid": "/m/0csby",
                        "description": "cloud",
                        "score": 0.97132415,
                        "topicality": 0.97132415
                }, {
                        "mid": "/m/01g5v",
                        "description": "blue",
                        "score": 0.9683707,
                        "topicality": 0.9683707
                }, {
                        "mid": "/m/02q7ylj",
                        "description": "daytime",
                        "score": 0.9555285,
                        "topicality": 0.9555285
                }, {
                        "mid": "/m/01ctsf",
                        "description": "atmosphere",
                        "score": 0.92822105,
                        "topicality": 0.92822105
                }, {
                        "mid": "/m/0csh5",
                        "description": "cumulus",
                        "score": 0.8386173,
                        "topicality": 0.8386173
                }, {
                        "mid": "/g/11k2xz7mr",
                        "description": "meteorological phenomenon",
                        "score": 0.75660443,
                        "topicality": 0.75660443
                }, {
                        "mid": "/m/026fm63",
                        "description": "calm",
                        "score": 0.72833425,
                        "topicality": 0.72833425
                }, {
                        "mid": "/m/03w43x",
                        "description": "computer wallpaper",
                        "score": 0.6601879,
                        "topicality": 0.6601879
                }, {
                        "mid": "/m/0d1n2",
                        "description": "horizon",
                        "score": 0.63659215,
                        "topicality": 0.63659215
                }]
        }]
}
```

Note that the Vision API does recognize it's an image with SKY and CLOUD but the type of cloud is incorrectly labeled as a cumulus cloud. We need a more specific model with our own labeled training data to get a more accurate model.

## Setup AutoML Vision

AutoML Vision enables you to train machine learning models to classify your images according to your own defined labels. In this section, we will upload images of clouds to Cloud Storage and use them to train a custom model to recognize different types of clouds (cumulus, cumulonimbus, etc.).

**Step 1**

In your GCP Console, click on the **Navigation menu** (![menu.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/2f22c727923782d61d44dd691c367624adaf6ddd6c33a5b6be24f43b3c913d3b.png)), click on **Vision**.

![72381e48433d551f.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/ef6a9cd8757e638c4b2d04499fa25b9c685ba3bd7697277cf031db076e228add.png)

**Step 2**

Select the GCP account created by qwiklabs (if prompted) and allow AutoML access :

![e1e349d894d056dd.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/d9e69848d9c32741fcb30a106f6757392b5f98aabdb66c31baccedd27833f21e.png)

**Step 3**

Click on **Get started with AutoML**.

![f728506dd99d1857.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/72a54b53642316ac76005046f8485694095057e8ba40fad8a9048ef81b59220b.png)

**Step 4**

Choose the correct GCP project created by qwiklabs and click **Continue**.

![e5d39cbdbb996a43.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/b686964e3fdf4351ab114590e121647ab446724d188910df079a7055e6f5a09e.png)

**Step 5**

Next, Click on **Go To Billing** and choose to **Go to linked billing account**:

![deb160242e65e406.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/63d42406477fcc7f74174bbc0f67a4b70ce36fe63fbc9dbfaca67895cd993bd6.png)

![3a40cb7562c52efc.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/169c7cbbc97c6c9d4e9c4a37a7a1c93df1cc2f34b628aebdbd234a8336a4c1ed.png)

**Step 6**

Confirm the step was successful.

![9816b5a6a0248310.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/fc166d6c461c1225952bac491e61a59bfbc098c80031acf9342fa6082e03b3d4.png)

**Step 7**

Now setup the necessary APIs and service accounts by clicking on **Set Up Now**.

![deb160242e65e406.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/63d42406477fcc7f74174bbc0f67a4b70ce36fe63fbc9dbfaca67895cd993bd6.png)

**Step 8**

You will be redirected on to the AutoML Vision console.

![d50aaa6165e60fef.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/d9a44896103efa1d6d554b4011e0ddc71fb0a7d9beaa0e5c05ccf1c46ef44b6c.png)

## Stage training files

**Step 1**

Back on your GCP console, check under storage buckets to confirm a new bucket created by AutoML Vision API. The name is similar to your project id, with the suffix *vcm* (for example : *qwiklabs-gcp-dabd0aa7da42381e-vcm*).

![9851fe6befb408d8.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/9d99708d4a8af373082f002ea41b4fdbfa7ee9749b314da77b59edde97b3bbd1.png)

Copy the new bucket name so you can use it in the next step.

**Step 2**

Set the bucket as an environment variable.

```bash
export BUCKET=<YOUR_AUTOML_BUCKET>
export BUCKET=qwiklabs-gcp-3fcd3c73127d895b-vcm
```

**Step 3**

Next, using the `gsutil` command line utility for Cloud Storage, copy the training images into your bucket:

```bash
gsutil -m cp -r gs://automl-codelab-clouds/* gs://${BUCKET}
```

After the copy, confirm that you have 3 folders in your storage bucket.

![acba0f20056c1e1d.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/fefbbe0fbdd8099ff3e63a86e4b12c3a9c8aad2afc84de4d4b3aa195e951a3f6.png)

## Create dataset

Now that your training data in Cloud Storage, you need a way for AutoML Vision to find them. To do this you'll create a CSV file where each row contains a URL to a training image and the associated label for that image. This CSV file has been created for you, you just need to update it with your bucket name.

**Step 1**

To do that, copy this file to your Cloud Shell instance:

```bash
gsutil cp gs://automl-codelab-metadata/data.csv .
```

**Step 2**

Then run the following command to update the CSV with the files in your project:

```bash
sed -i -e "s/placeholder/${BUCKET}/g" ./data.csv
```

```
$ cat data.csv
gs://qwiklabs-gcp-3fcd3c73127d895b-vcm/cirrus/1.jpg,cirrus
gs://qwiklabs-gcp-3fcd3c73127d895b-vcm/cirrus/10.jpg,cirrus
[...]
gs://qwiklabs-gcp-3fcd3c73127d895b-vcm/cirrus/9.jpg,cirrus
gs://qwiklabs-gcp-3fcd3c73127d895b-vcm/cumulonimbus/1.jpg,cumulonimbus
[...]
gs://qwiklabs-gcp-3fcd3c73127d895b-vcm/cumulonimbus/9.jpg,cumulonimbus
gs://qwiklabs-gcp-3fcd3c73127d895b-vcm/cumulus/1.jpg,cumulus
gs://qwiklabs-gcp-3fcd3c73127d895b-vcm/cumulus/10.jpg,cumulus
[...]
gs://qwiklabs-gcp-3fcd3c73127d895b-vcm/cumulus/9.jpg,cumulus
```



**Step 3**

Now you're ready to upload this file to your Cloud Storage bucket:

```bash
gsutil cp ./data.csv gs://${BUCKET}
```

Confirm that you see the CSV file in your bucket.

**Step 4**

Navigate back to the [AutoML Vision UI](https://cloud.google.com/automl/ui/vision).

![c1d8628f395e3753.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/223a6c863315a4d026bd488b82c1c4a2865da58b48fbb6dca4d22ad6624399c1.png)

**Note:** If you've previously created a dataset with AutoML vision, you will see a list of datasets instead. In this case, click **+ New Dataset**.

Type "clouds" for the Dataset name.

Choose **Select a CSV file on Cloud Storage** and enter the URL of the file you just uploaded - `gs://your-project-name-vcm/data.csv`

![cd7d937018a32bc3.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/3a0e71c731f0e25fdbb0f94ba992f4d828906adbb4f7d76fefb4888af2aa1204.png)

For this example, leave "enable multi-label classification" unchecked. In your own projects, you may want to check this box if you want to assign multiple labels per image.

![e8ff1ff07b0a1b94.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/c81f66202fb9578c43e24fb21650cc8dd843acdcdde004d4713390bbff214b59.png)

Select **Create Dataset**.

![9526284aab049bcc.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/c6743f39919f3cd031200ff426f90893c2dd98ea9b0ebb6c5d35592e133dede2.png)

It will take around 2 minutes for your images to finish importing. Once the import has completed, you'll be brought to a page with all the images in your dataset.

## Inspect images

**Step 1**

After the import completes, you will see the Images tab.

![cf21b76c90b91093.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/83ac2b90b631dca6517a1184d45d8b12d7428277eff002c853c0ebb0a3bc3c73.png)

Try filtering by different labels (i.e. click cumulus) to review the training images:

![44f615aeaccfdd8e.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/2422cad4b257dacf18b3a17587e53cb24fdfcdb33b30777a1aa1af9370162fd5.png)

**Note:** If you were building a production model, you'd want *at least*100 images per label to ensure high accuracy. This is just a demo so we only used 20 images so that our model will train quickly.

**Step 2**

If any image is labeled incorrectly you can click on them to switch the label or delete the image from your training set:

![5eb5e78a8ac730c2.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/7d73aab38284515f8c918ba3bcfb7d9e514692458b2af2cd06a5e80d01edb22c.png)

To see a summary of how many images you have for each label, click on Label stats. You should see the following show up on the left side of your browser.

![label_stats.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/48d598197912106a0da77b2550df583fee7b3d63c2474e4b5bab68c9a10f071d.png)

**Note:** If you are working with a dataset that isn't already labeled, AutoML Vision provides an in-house human labeling service.

## Train your model

You're ready to start training your model! AutoML Vision handles this for you automatically, without requiring you to write any of the model code.

**Step 1**

To train your clouds model, go to the **Train** tab and click **Start Training**.

Enter a name for your model, or use the default auto-generated name, and click **Start Training**.

![1fecfd23c9862dbe.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/c1859ea922e8b1decf068a2290be911a1079877130d6dd6673c8e7f22dffa155.png)

![85f23629111b37e6.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/0f0ad60a96fd8ef7f8bcba6315998ee6feadca123d6d2ad75d9ced5a6ebccfbe.png)

Since this is a small dataset, it will only take around **5 minutes** to complete.

## Evaluate your model

**Step 1**

In the **Evaluate** tab, you'll see information about AUC, precision and recall of the model.

![evaluate.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/c2a53caa6090106bc4cdde991970ef2f1e5a34f5bc68c48a98a33cef110eb5bb.png)

You can also play around with **Score threshold**:

![score_threshold.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/a12a9987bf6f436afe53b7f65766ccbe3de2e57cf529a5306add30f12203ffd0.png)

Finally, scroll down to take a look at the **Confusion matrix**.

![confusion_matrix.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/0d102e57f01a3e480d48b4fafa521074f025d843d4ab1cfb9651d1415d756890.png)

All of this provides some common machine learning metrics to evaluate your model accuracy and see where you can improve your training data. Since the focus for this lab was not on accuracy, skip to the prediction section, but feel free to browse the accuracy metrics on your own.

## Generate predictions

Now it's time for the most important part: generating predictions on your trained model using data it hasn't seen before.

**Step 1**

Navigate to the **Predict** tab in the AutoML UI:

![8c40ba3f466f0af4.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/3086ee8ec64d228c46f42d1c1a94accdda33ce75aef5b0fc0994b5213d3fff96.png)

There are a few ways to generate predictions. In this lab, you'll use the UI to upload images. You'll see how your model does classifying these two images (the first is a cirrus cloud, the second is a cumulonimbus).

**Step 2**

Download these images by right-clicking on each of them:

![a4e6d50183e83703.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/376a6cca994cde414af4d1238e96a4dc23c8861f08bab63b4e7f6fab38b8afc3.jpg)

![1d4aaa17ec62e9ba.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/19994146600a1b3b03a13f323510a1e959b17e5c4b11012488a3e8858c236bde.jpg)

**Step 3**

Return to the UI, select **upload images** and upload them to the online prediction UI. When the prediction request completes you should see something like the following:

![prediction.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/254db06b4e092f44d8cbe880b0f028feae91012a707937fcbc50c95bfeae700a.png)

Pretty cool - the model classified each type of cloud correctly! Does your trained model do better than the 57% CIRRUS cloud above?

**Note:** In addition to generating predictions in the AutoML UI, you can also use the REST API or the Python client to make prediction requests on your trained model. Check out the tabs for each to see some sample code. You can try it out by copy/pasting these commands into Cloud Shell and providing an image URL.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.







# [[eof]]



[[todo]]:

* make python and html files of all **notebooks** in all courses!
* make pdfs of all **.md-files i**n all courses! (add to .gitignore)

