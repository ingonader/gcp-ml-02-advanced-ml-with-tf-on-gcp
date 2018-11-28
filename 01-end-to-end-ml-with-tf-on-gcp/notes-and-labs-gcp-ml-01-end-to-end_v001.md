# End-to-End Machine Learning with TensorFlow on GCP



## to download materials

software to download materials: <https://github.com/coursera-dl/coursera-dl>

link to coursera material: <https://www.coursera.org/learn/end-to-end-ml-tensorflow-gcp/lecture/wWBkE/effective-ml>

```bash
coursera-dl -u ingo.nader@gmail.com end-to-end-ml-tensorflow-gcp --download-quizzes --download-notebooks --about
```



## Notes

* Do labs in Google Chrome (recommended)
* Open Incognito window, go to <http://console.cloud.google.com/>



### Add stuff to git

* SSH into cloudvm instance

  * either by selecting the SSH button in the VM instances list (GCP sandwich menu)

  * or by running

    ```bash
    gcloud compute --project project-id ssh \
      --zone zone instance-name
    ```

* then execute an interactive bash session in the docker container that runs the lab:

  ```bash
  sudo docker ps
  docker exec -it <container-id> bash
  cd content/datalab/
  ```





# End to End ML Lab 1: Explore dataset

## Overview

*Duration is 1 min*

This lab is part of a lab series where you train, evaluate, and deploy a machine learning model to predict a baby's weight.

In this lab #1, you explore and visualize a BigQuery dataset

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80678/original/img/2179c12ec7ea1f2b.png)

### **What you learn**

In this lab, you will learn to:

* Explore the natality dataset in BigQuery using Datalab
* Use Python package to execute the query and convert the result into a Pandas dataframe for visualization.

## Setup

For each lab, you get a new GCP project and set of resources for a fixed time at no cost.

Note the lab's access time (for example, 02:00:00) and make sure you can finish in that time block.

There isn't a pause feature, so the clock keeps running. You can start again if needed.

1. When ready, click **Start Lab.**
2. Note your lab credentials. You will use them to sign in to Cloud Platform Console.![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80678/original/img/eb2f9b843e5c184c.png)
3. Click **Open Google Console**.
4. Click **Use another account** if displayed.

**IMPORTANT :** Only use lab credentials for **THIS**lab. Do **NOT** use previous lab credentials or personal Google or Gmail credentials. Using previous lab credentials generates a permission error. Using personal Google or Gmail credentials **incurs charges**.

1. Copy and paste your Qwiklabs username and password for this lab into the prompts.
2. Accept the terms and skip the recovery resource page.

**IMPORTANT :** Do not click **End** unless you are finished with the lab. This clears your work and removes the project. It's not a pause button.

## Launch Cloud Datalab

To launch Cloud Datalab:

### **Step 1**

Open Cloud Shell. The cloud shell icon is at the top right of the Google Cloud Platform [web console](https://console.cloud.google.com/):

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

```bash
datalab create mydatalabvm --zone europe-west1-c
```



-Replace <ZONE> with a zone name you picked from the previous step.

Note: follow the prompts during this process.

Datalab will take about 5 minutes to start.

### **Step 4**

Look back at Cloud Shell, and follow any prompts. If asked for a ssh passphrase, just hit return (for no passphrase).

### **Step 5**

If necessary, wait for Datalab to finish launching. Datalab is ready when you see a message prompting you to do a "Web Preview".

### **Step 6**

Click on the **Web Preview** icon on the top-right corner of the Cloud Shell ribbon. Switch or enter the port **8081**.

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80678/original/img/7eb159ad9b4d3d2d.png)

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80678/original/img/f448f8cfb1a15a7e.png)

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell.

## Clone course repo within your Datalab instance

To clone the course repo in your datalab instance:

### **Step 1**

In Cloud Datalab home page (browser), navigate into "**notebooks**" and add a new notebook using the icon ![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80678/original/img/5fdee4bbcdee4b9a.png) on the top left.

### **Step 2**

Rename this notebook as ‘**repocheckout**'.

### **Step 3**

In the new notebook, enter the following commands in the cell, and click on **Run** (on the top navigation bar) to run the commands:

```
%bash
git clone https://github.com/GoogleCloudPlatform/training-data-analyst
rm -rf training-data-analyst/.git
```

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80678/original/img/1a3d26cf1e7e1e5f.png)

### **Step 4**

Confirm that you have cloned the repo by going back to Datalab browser, and ensure you see the **training-data-analyst** directory. All the files for all labs throughout this course are available in this directory.

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80678/original/img/c165e0bdef4a0ecc.png)

## Explore dataset

*Duration is 15 min*

Explore a BigQuery dataset to find features to use in an ML model.

### **Step 1**

In Cloud Datalab, click on the Home icon, and then navigate to **notebooks/training-data-analyst > courses > machine_learning > deepdive > 06_structured > labs**and open **1_explore.ipynb**.

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

### **Step 2**

In Datalab, click on **Clear | All Cells**. Now read the narrative and execute each cell in turn:

* If you notice sections marked "Lab Task", you will need to create a new code cell and write/complete code to achieve the task.
* Some lab tasks include starter code. In such cells, look for lines marked #TODO.
* Hints may also be provided for the tasks to guide you along. Highlight the text to read the hints (they are in white text).
* If you need more help, you may take a look at the complete solution by navigating to : **notebooks/training-data-analyst > courses > machine_learning > deepdive > 06_structured** and open **1_explore.ipynb**

Note: when doing copy/paste of python code, please be careful about indentation

©Google, Inc. or its affiliates. All rights reserved. Do not distribute.



# End to End ML Lab 2: Create a sample dataset

## Overview

*Duration is 1 min*

This lab is part of a lab series where you train, evaluate, and deploy a machine learning model to predict a baby's weight.

In this lab #2, you sample the full BigQuery dataset to create a smaller dataset for model development and local training.

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80679/original/img/3862a7e90f775ad6.png)

### **What you learn**

In this lab, you will learn how to to:

* Sample a BigQuery dataset to create datasets for ML
* Preprocess data using Pandas

## Setup

For each lab, you get a new GCP project and set of resources for a fixed time at no cost.

Note the lab's access time (for example, 02:00:00) and make sure you can finish in that time block.

There isn't a pause feature, so the clock keeps running. You can start again if needed.

1. When ready, click **Start Lab.**
2. Note your lab credentials. You will use them to sign in to Cloud Platform Console.![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80679/original/img/eb2f9b843e5c184c.png)
3. Click **Open Google Console**.
4. Click **Use another account** if displayed.

**IMPORTANT :** Only use lab credentials for **THIS** lab. Do **NOT** use previous lab credentials or personal Google or Gmail credentials. Using previous lab credentials generates a permission error. Using personal Google or Gmail credentials **incurs charges**.

1. Copy and paste your Qwiklabs username and password for this lab into the prompts.
2. Accept the terms and skip the recovery resource page.

**IMPORTANT :** Do not click **End** unless you are finished with the lab. This clears your work and removes the project. It's not a pause button.

## Launch Cloud Datalab

To launch Cloud Datalab:

### **Step 1**

Open Cloud Shell. The cloud shell icon is at the top right of the Google Cloud Platform [web console](https://console.cloud.google.com/):

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

Click on the **Web Preview** icon on the top-right corner of the Cloud Shell ribbon. Switch or enter the port **8081**.

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80679/original/img/7eb159ad9b4d3d2d.png)

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80679/original/img/f448f8cfb1a15a7e.png)

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell.

## Clone course repo within your Datalab instance

To clone the course repo in your datalab instance:

### **Step 1**

In Cloud Datalab home page (browser), navigate into "**notebooks**" and add a new notebook using the icon ![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80679/original/img/5fdee4bbcdee4b9a.png) on the top left.

### **Step 2**

Rename this notebook as ‘**repocheckout**'.

### **Step 3**

In the new notebook, enter the following commands in the cell, and click on **Run**(on the top navigation bar) to run the commands:

```
%bash
git clone https://github.com/GoogleCloudPlatform/training-data-analyst
rm -rf training-data-analyst/.git
```

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80679/original/img/1a3d26cf1e7e1e5f.png)

### **Step 4**

Confirm that you have cloned the repo by going back to Datalab browser, and ensure you see the **training-data-analyst** directory. All the files for all labs throughout this course are available in this directory.

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80679/original/img/c165e0bdef4a0ecc.png)

## Create sampled dataset

*Duration is 15 min*

Sample a BigQuery dataset to create datasets for ML

### **Step 1**

In Cloud Datalab, click on the Home icon, and then navigate to **notebooks/training-data-analyst > courses > machine_learning > deepdive > 06_structured > labs** and open **2_sample.ipynb**.

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

### **Step 2**

In Datalab, click on **Clear | All Cells**. Now read the narrative and execute each cell in turn:

* If you notice sections marked "Lab Task", you will need to create a new code cell and write/complete code to achieve the task.
* Some lab tasks include starter code. In such cells, look for lines marked #TODO.
* Hints may also be provided for the tasks to guide you along. Highlight the text to read the hints (they are in white text).
* If you need more help, you may take a look at the complete solution by navigating to : **notebooks/training-data-analyst > courses > machine_learning > deepdive > 06_structured** and open **2_sample.ipynb**

Note: when doing copy/paste of python code, please be careful about indentation

©Google, Inc. or its affiliates. All rights reserved. Do not distribute.



# End to End ML Lab 3: Create TensorFlow model

## Overview

*Duration is 1 min*

This lab is part of a lab series where you train, evaluate, and deploy a machine learning model to predict a baby's weight.  

In this lab #3, you create a TensorFlow model using the high-level Estimator API.

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80680/original/img/78d97c5a4911535d.png)

### **What you learn**

In this lab, you will learn how to:

* Use the Estimator API to build a linear model
* Use the Estimator API to build wide and deep model
* Monitor training using TensorBoard

## Setup

For each lab, you get a new GCP project and set of resources for a fixed time at no cost.

Note the lab's access time (for example, 02:00:00) and make sure you can finish in that time block.

There isn't a pause feature, so the clock keeps running. You can start again if needed.

1. When ready, click **Start Lab.** 
2. Note your lab credentials. You will use them to sign in to Cloud Platform Console.![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80680/original/img/eb2f9b843e5c184c.png) 
3. Click **Open Google Console**.
4. Click **Use another account** if displayed.

**IMPORTANT :** Only use lab credentials for **THIS** lab.  Do **NOT**  use previous lab credentials or personal Google or Gmail credentials.  Using previous lab credentials generates a permission error.  Using  personal Google or Gmail credentials **incurs charges**.

1. Copy and paste your Qwiklabs username and password for this lab into the prompts.
2. Accept the terms and skip the recovery resource page.

**IMPORTANT :** Do not click **End** unless you are finished with the lab. This clears your work and removes the project. It's not a pause button.

## Launch Cloud Datalab

To launch Cloud Datalab:

### **Step 1**

Open Cloud Shell. The cloud shell icon is at the top right of the Google Cloud Platform [web console](https://console.cloud.google.com/):

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

Click on the **Web Preview** icon on the top-right corner of the Cloud Shell ribbon.  Switch or enter the port **8081**.

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80680/original/img/7eb159ad9b4d3d2d.png)

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80680/original/img/f448f8cfb1a15a7e.png)

Note: If the cloud shell used for running the  datalab command is closed or interrupted, the connection to your Cloud  Datalab VM will terminate. If that happens, you may be able to reconnect  using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. 

## Clone course repo within your Datalab instance

To clone the course repo in your datalab instance:

### **Step 1**

In Cloud Datalab home page (browser), navigate into "**notebooks**" and add a new notebook using the icon ![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80680/original/img/5fdee4bbcdee4b9a.png)  on the top left.

### **Step 2**

Rename this notebook as ‘**repocheckout**'.

### **Step 3**

In the new notebook, enter the following commands in the cell, and click on **Run** (on the top navigation bar) to run the commands:

```
%bash
git clone https://github.com/GoogleCloudPlatform/training-data-analyst
rm -rf training-data-analyst/.git
```

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80680/original/img/1a3d26cf1e7e1e5f.png)

### **Step 4**

Confirm that you have cloned the repo by going back to Datalab browser, and ensure you see the **training-data-analyst** directory. All the files for all labs throughout this course are available in this directory. 

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80680/original/img/c165e0bdef4a0ecc.png)

## Create TF model

*Duration is 15 min*

Use the Estimator API to create a TensorFlow model

### **Step 1**

In Cloud Datalab, click on the Home icon, and then navigate to **notebooks/training-data-analyst > courses > machine_learning > deepdive > 06_structured > labs** and open **3_tensorflow.ipynb**.

Note: If the cloud shell used for running the  datalab command is closed or interrupted, the connection to your Cloud  Datalab VM will terminate. If that happens, you may be able to reconnect  using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

### **Step 2**

In Datalab, click on **Clear | All Cells**. Now read the narrative and execute each cell in turn:

* If you notice sections marked "Lab Task", you will need to create a new code cell and write/complete code to achieve the task.
* Some lab tasks include starter code. In such cells, look for lines marked #TODO. 
* Hints may also be provided for the tasks to guide you along. Highlight the text to read the hints (they are in white text).
*  If you need more help, you may take a look at the complete solution by navigating to : **notebooks/training-data-analyst > courses > machine_learning > deepdive > 06_structured**  and open one of the 2 notebooks: **3_tensorflow_dnn.ipynb** or **3_tensorflow_wd.ipynb** 

Note: when doing copy/paste of python code, please be careful about indentation

©Google, Inc. or its affiliates. All rights reserved. Do not distribute.





# End to End ML Lab 4 : Preprocessing using Dataflow

## Overview

*Duration is 1 min*

This lab is part of a lab series where you train, evaluate, and deploy a machine learning model to predict a baby's weight.  

In this lab #4, you use Dataflow to create datasets for Machine Learning

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80681/original/img/46e68f1273b92e2d.png)

### **What you learn**

In this lab, you will learn how to:

* Create ML dataset using Dataflow

## Setup

For each lab, you get a new GCP project and set of resources for a fixed time at no cost.

Note the lab's access time (for example, 02:00:00) and make sure you can finish in that time block.

There isn't a pause feature, so the clock keeps running. You can start again if needed.

1. When ready, click **Start Lab.** 
2. Note your lab credentials. You will use them to sign in to Cloud Platform Console.![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80681/original/img/eb2f9b843e5c184c.png) 
3. Click **Open Google Console**.
4. Click **Use another account** if displayed.

**IMPORTANT :** Only use lab credentials for **THIS** lab.  Do **NOT**  use previous lab credentials or personal Google or Gmail credentials.  Using previous lab credentials generates a permission error.  Using  personal Google or Gmail credentials **incurs charges**.

1. Copy and paste your Qwiklabs username and password for this lab into the prompts.
2. Accept the terms and skip the recovery resource page.

**IMPORTANT :** Do not click **End** unless you are finished with the lab. This clears your work and removes the project. It's not a pause button.

## Launch Cloud Datalab

To launch Cloud Datalab:

### **Step 1**

Open Cloud Shell. The cloud shell icon is at the top right of the Google Cloud Platform [web console](https://console.cloud.google.com/):

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

Click on the **Web Preview** icon on the top-right corner of the Cloud Shell ribbon.  Switch or enter the port **8081**.

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80681/original/img/7eb159ad9b4d3d2d.png)

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80681/original/img/f448f8cfb1a15a7e.png)

Note: If the cloud shell used for running the  datalab command is closed or interrupted, the connection to your Cloud  Datalab VM will terminate. If that happens, you may be able to reconnect  using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. 

## Clone course repo within your Datalab instance

To clone the course repo in your datalab instance:

### **Step 1**

In Cloud Datalab home page (browser), navigate into "**notebooks**" and add a new notebook using the icon ![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80681/original/img/5fdee4bbcdee4b9a.png)  on the top left.

### **Step 2**

Rename this notebook as ‘**repocheckout**'.

### **Step 3**

In the new notebook, enter the following commands in the cell, and click on **Run** (on the top navigation bar) to run the commands:

```
%bash
git clone https://github.com/GoogleCloudPlatform/training-data-analyst
rm -rf training-data-analyst/.git
```

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80681/original/img/1a3d26cf1e7e1e5f.png)

### **Step 4**

Confirm that you have cloned the repo by going back to Datalab browser, and ensure you see the **training-data-analyst** directory. All the files for all labs throughout this course are available in this directory. 

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80681/original/img/c165e0bdef4a0ecc.png)

## Create ML dataset using Dataflow

*Duration is 15 min*

### **Step 1**

In Cloud Datalab, click on the Home icon, and then navigate to **notebooks/training-data-analyst > courses > machine_learning > deepdive > 06_structured > labs** and open **4_preproc.ipynb**.

Note: If the cloud shell used for running the  datalab command is closed or interrupted, the connection to your Cloud  Datalab VM will terminate. If that happens, you may be able to reconnect  using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

### **Step 2**

In Datalab, click on **Clear | All Cells**. Now read the narrative and execute each cell in turn:

* If you notice sections marked "Lab Task", you will need to create a new code cell and write/complete code to achieve the task.
* Some lab tasks include starter code. In such cells, look for lines marked #TODO. 
* Hints may also be provided for the tasks to guide you along. Highlight the text to read the hints (they are in white text).
*  If you need more help, you may take a look at the complete solution by navigating to : **notebooks/training-data-analyst > courses > machine_learning > deepdive > 06_structured**  and open: **4_preproc.ipynb**  

Note: when doing copy/paste of python code, please be careful about indentation

©Google, Inc. or its affiliates. All rights reserved. Do not distribute.



# End to End ML Lab 5 : Training on Cloud ML Engine

## Overview

*Duration is 1 min*

This lab is part of a lab series where you train, evaluate, and deploy a machine learning model to predict a baby's weight.  

In this lab #5, you will do distributed training and hyperparameter tuning on Cloud ML Engine.

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80682/original/img/22386898f5514bc5.png)

### **What you learn**

In this lab, you will learn how to:

* do distributed training using Cloud ML Engine
* Improve model accuracy using hyperparameter tuning

## Setup

For each lab, you get a new GCP project and set of resources for a fixed time at no cost.

Note the lab's access time (for example, 02:00:00) and make sure you can finish in that time block.

There isn't a pause feature, so the clock keeps running. You can start again if needed.

1. When ready, click **Start Lab.** 
2. Note your lab credentials. You will use them to sign in to Cloud Platform Console.![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80682/original/img/eb2f9b843e5c184c.png) 
3. Click **Open Google Console**.
4. Click **Use another account** if displayed.

**IMPORTANT :** Only use lab credentials for **THIS** lab.  Do **NOT**  use previous lab credentials or personal Google or Gmail credentials.  Using previous lab credentials generates a permission error.  Using  personal Google or Gmail credentials **incurs charges**.

1. Copy and paste your Qwiklabs username and password for this lab into the prompts.
2. Accept the terms and skip the recovery resource page.

**IMPORTANT :** Do not click **End** unless you are finished with the lab. This clears your work and removes the project. It's not a pause button.

## Launch Cloud Datalab

To launch Cloud Datalab:

### **Step 1**

Open Cloud Shell. The cloud shell icon is at the top right of the Google Cloud Platform [web console](https://console.cloud.google.com/):

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

Click on the **Web Preview** icon on the top-right corner of the Cloud Shell ribbon.  Switch or enter the port **8081**.

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80682/original/img/7eb159ad9b4d3d2d.png)

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80682/original/img/f448f8cfb1a15a7e.png)

Note: If the cloud shell used for running the  datalab command is closed or interrupted, the connection to your Cloud  Datalab VM will terminate. If that happens, you may be able to reconnect  using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. 

## Clone course repo within your Datalab instance

To clone the course repo in your datalab instance:

### **Step 1**

In Cloud Datalab home page (browser), navigate into "**notebooks**" and add a new notebook using the icon ![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80682/original/img/5fdee4bbcdee4b9a.png)  on the top left.

### **Step 2**

Rename this notebook as ‘**repocheckout**'.

### **Step 3**

In the new notebook, enter the following commands in the cell, and click on **Run** (on the top navigation bar) to run the commands:

```
%bash
git clone https://github.com/GoogleCloudPlatform/training-data-analyst
rm -rf training-data-analyst/.git
```

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80682/original/img/1a3d26cf1e7e1e5f.png)

### **Step 4**

Confirm that you have cloned the repo by going back to Datalab browser, and ensure you see the **training-data-analyst** directory. All the files for all labs throughout this course are available in this directory. 

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80682/original/img/c165e0bdef4a0ecc.png)

## Training on Cloud ML Engine

*Duration is 15 min*

### **Step 1**

In Cloud Datalab, click on the Home icon, and then navigate to **notebooks/training-data-analyst > courses > machine_learning > deepdive > 06_structured > labs** and open **5_train.ipynb**.

Note: If the cloud shell used for running the  datalab command is closed or interrupted, the connection to your Cloud  Datalab VM will terminate. If that happens, you may be able to reconnect  using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

### **Step 2**

In Datalab, click on **Clear | All Cells**. Now read the narrative and execute each cell in turn:

* If you notice sections marked "Lab Task", you will need to create a new code cell and write/complete code to achieve the task.
* Some lab tasks include starter code. In such cells, look for lines marked #TODO. 
* Hints may also be provided for the tasks to guide you along. Highlight the text to read the hints (they are in white text).
*  If you need more help, you may take a look at the complete solution by navigating to : **notebooks/training-data-analyst > courses > machine_learning > deepdive > 06_structured**  and open: **5_train.ipynb**  

Note: when doing copy/paste of python code, please be careful about indentation

©Google, Inc. or its affiliates. All rights reserved. Do not distribute.

# End to End ML Lab 5b BQML (optional): Predict Baby Weight using BQML (beta)

## Overview

BigQuery is Google's fully managed, NoOps, low cost analytics  database. With BigQuery you can query terabytes and terabytes of data  without having any infrastructure to manage or needing a database  administrator. BigQuery uses SQL and can take advantage of the  pay-as-you-go model. BigQuery allows you to focus on analyzing data to  find meaningful insights.

[BigQuery Machine Learning](https://cloud.google.com/bigquery/docs/bigqueryml-analyst-start)  (BQML, product in beta) is a new feature in BigQuery where data  analysts can create, train, evaluate, and predict with machine learning  models with minimal coding.

### Objectives

In this lab, you learn to perform the following tasks:

* Use BigQuery to explore the natality dataset
* Create a training and evaluation dataset for prediction
* Create a regression (linear regression) model in BQML
* Evaluate the performance of your machine learning model
* Use feature engineering to improve model accuracy
* Predict baby weight from a set of features

## Introduction

In this lab, you will be using the CDC's natality data to build a  model to predict baby weights based on a handful of features known at  pregnancy. Because we're predicting a continuous value, this is a  regression problem, and for that, we'll use the linear regression model  built into BQML.

## Setup

#### What you'll need

To complete this lab, you’ll need:

* Access to a standard internet browser (Chrome browser recommended).
* Time. Note the lab’s **Completion** time in Qwiklabs.  This is an estimate of the time it should take to complete all steps.   Plan your schedule so you have time to complete the lab. Once you start  the lab, you will not be able to pause and return later (you begin at  step 1 every time you start a lab).
* The lab's **Access** time is how long your lab resources  will be available. If you finish your lab with access time still  available, you will be able to explore the Google Cloud Platform or work  on any section of the lab that was marked "if you have time". Once the  Access time runs out, your lab will end and all resources will  terminate.
* You **DO NOT** need a Google Cloud Platform account or  project. An account, project and associated resources are provided to  you as part of this lab.
* If you already have your own GCP account, make sure you do not use it for this lab.
* If your lab prompts you to log into the console, **use only the student account provided to you by the lab**. This prevents you from incurring charges for lab activities in your personal GCP account.

#### Start your lab

When you are ready, click **Start Lab**. You can track your lab’s progress with the status bar at the top of your screen.

 **Important** What is happening during this time?   Your lab is spinning up GCP resources for you behind the scenes,  including an account, a project, resources within the project, and  permission for you to control the resources needed to run the lab. This  means that instead of spending time manually setting up a project and  building resources from scratch as part of your lab, you can begin  learning more quickly.  

#### Find Your Lab’s GCP Username and Password

To access the resources and console for this lab, locate the  Connection Details panel in Qwiklabs.  Here you will find the account ID  and password for the account you will use to log in to the Google Cloud  Platform:

![Open Google Console](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/5eaee037e6cedbf49f6a702ab8a9ef820bb8e717332946ff76a0831a6396aafc.png)

If your lab provides other resource identifiers or connection-related information, it will appear on this panel as well.

### Activate Google Cloud Shell

Google Cloud Shell provides command-line access to your GCP resources.

From the GCP Console click the **Cloud Shell** icon on the top right toolbar:

![Cloud Shell Icon](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/718029dee0e562c61c14536c5a636a5bae0ef5136e9863b98160d1e06123908a.png)

Then click **START CLOUD SHELL**:

![Start Cloud Shell](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/feb5ea74b4a4f6dfac7800f39c3550364ed7a33a7ab17b6eb47cab3e65c33b13.png)

 You can click **START CLOUD SHELL** immediately when the dialog comes up instead of waiting in the dialog until the Cloud Shell provisions.  

It  takes a few moments to provision and connects to the environment:

![Cloud Shell Terminal](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/11def2e8f4cfd6f1590f3fd825d4566658501ca87e1d5d1552aa17339050c194.png)

The Cloud Shell is a virtual machine loaded with all the development  tools you’ll need. It offers a persistent 5GB home directory, and runs  on the Google Cloud, greatly enhancing network performance and  authentication.

Once connected to the cloud shell, you'll see that you are already authenticated and  the project is set to your *PROJECT_ID*:

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

   Note: The connection to your Datalab instance remains open for as  long as the datalab command is active. If the cloud shell used for  running the datalab command is closed or interrupted, the connection to  your Cloud Datalab VM will terminate. If that happens, you may be able  to reconnect using the command `datalab connect bdmlvm` in your new Cloud Shell.

To clone the course repo in your datalab instance:

**Step 1**

In Cloud Datalab home page (browser), navigate into **“notebooks”** and add a new notebook using the icon ![notebook.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/fef0cc8c36a1856aa4ca73423f2ba59dde635267437c1253c268f366dfe19899.png)  on the top left.

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

1. In the Datalab browser, navigate to **training-data-analyst > courses > machine_learning > deepdive > 06_structured > 5_train_bqml.ipynb**
2. Read the commentary, **Click Clear | Clear all Cells**, then run the Python snippets (Use **Shift+Enter** to run each piece of code) in the cell, step by step.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.

You will be given an opportunity to rate the lab experience. Select  the applicable number of stars, type a comment, and then click **Submit**.

The number of stars indicates the following:

* 1 star = Very dissatisfied
* 2 stars = Dissatisfied
* 3 stars = Neutral
* 4 stars = Satisfied
* 5 stars = Very satisfied

You can close the dialog box if you don't want to provide feedback.

For feedback, suggestions, or corrections, please use the **Support** tab.

©Google, Inc. or its affiliates. All rights reserved. Do not distribute.



# End to End ML Lab 6 : Deploying and Predicting with Cloud ML Engine

## Overview

*Duration is 1 min*

This lab is part of a lab series where you train, evaluate, and deploy a machine learning model to predict a baby's weight.  

In this lab #6, you deploy a trained model to Cloud ML Engine and send requests to it to get baby weight predictions.

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80683/original/img/a336a0047f32fe2d.png)

### **What you learn**

In this lab, you will learn how to:

* deploy the trained model to act as a REST web service
* send a JSON request to the endpoint of the service to make it predict a baby's weight. 

## Setup

For each lab, you get a new GCP project and set of resources for a fixed time at no cost.

Note the lab's access time (for example, 02:00:00) and make sure you can finish in that time block.

There isn't a pause feature, so the clock keeps running. You can start again if needed.

1. When ready, click **Start Lab.** 
2. Note your lab credentials. You will use them to sign in to Cloud Platform Console.![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80683/original/img/eb2f9b843e5c184c.png) 
3. Click **Open Google Console**.
4. Click **Use another account** if displayed.

**IMPORTANT :** Only use lab credentials for **THIS** lab.  Do **NOT**  use previous lab credentials or personal Google or Gmail credentials.  Using previous lab credentials generates a permission error.  Using  personal Google or Gmail credentials **incurs charges**.

1. Copy and paste your Qwiklabs username and password for this lab into the prompts.
2. Accept the terms and skip the recovery resource page.

**IMPORTANT :** Do not click **End** unless you are finished with the lab. This clears your work and removes the project. It's not a pause button.

## Launch Cloud Datalab

To launch Cloud Datalab:

### **Step 1**

Open Cloud Shell. The cloud shell icon is at the top right of the Google Cloud Platform [web console](https://console.cloud.google.com/):

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

Click on the **Web Preview** icon on the top-right corner of the Cloud Shell ribbon.  Switch or enter the port **8081**.

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80683/original/img/7eb159ad9b4d3d2d.png)

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80683/original/img/f448f8cfb1a15a7e.png)

Note: If the cloud shell used for running the  datalab command is closed or interrupted, the connection to your Cloud  Datalab VM will terminate. If that happens, you may be able to reconnect  using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. 

## Clone course repo within your Datalab instance

To clone the course repo in your datalab instance:

### **Step 1**

In Cloud Datalab home page (browser), navigate into "**notebooks**" and add a new notebook using the icon ![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80683/original/img/5fdee4bbcdee4b9a.png)  on the top left.

### **Step 2**

Rename this notebook as ‘**repocheckout**'.

### **Step 3**

In the new notebook, enter the following commands in the cell, and click on **Run** (on the top navigation bar) to run the commands:

```
%bash
git clone https://github.com/GoogleCloudPlatform/training-data-analyst
rm -rf training-data-analyst/.git
```

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80683/original/img/1a3d26cf1e7e1e5f.png)

### **Step 4**

Confirm that you have cloned the repo by going back to Datalab browser, and ensure you see the **training-data-analyst** directory. All the files for all labs throughout this course are available in this directory. 

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80683/original/img/c165e0bdef4a0ecc.png)

## Deploy and predict with model

*Duration is 15 min*

### **Step 1**

In Cloud Datalab, click on the Home icon, and then navigate to **notebooks/training-data-analyst > courses > machine_learning > deepdive > 06_structured > labs** and open **6_deploy.ipynb**.

Note: If the cloud shell used for running the  datalab command is closed or interrupted, the connection to your Cloud  Datalab VM will terminate. If that happens, you may be able to reconnect  using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

### **Step 2**

In Datalab, click on **Clear | All Cells**. Now read the narrative and execute each cell in turn:

* If you notice sections marked "Lab Task", you will need to create a new code cell and write/complete code to achieve the task.
* Some lab tasks include starter code. In such cells, look for lines marked #TODO. 
* Hints may also be provided for the tasks to guide you along. Highlight the text to read the hints (they are in white text).
*  If you need more help, you may take a look at the complete solution by navigating to : **notebooks/training-data-analyst > courses > machine_learning > deepdive > 06_structured**  and open: **6_deploy.ipynb**  

Note: when doing copy/paste of python code, please be careful about indentation

©Google, Inc. or its affiliates. All rights reserved. Do not distribute.



# End to End ML Lab 7 :  Building an App Engine app to serve ML predictions

## Overview

*Duration is 1 min*

This lab is part of a lab series where you train, evaluate, and deploy a machine learning model to predict a baby's weight.  

In this lab #7, you build an app using Google App Engine with Flask.  This will provide a front-end that will allow end users to interactively  receive predictions from the deployed model.

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80684/original/img/d1e80885c2eec73c.png)

### **What you learn**

In this lab, you will learn how to:

* Deploy a python Flask app as a App Engine web application
* Use the App Engine app to post JSON data, based on user interface input, to the deployed ML model and get predictions

## Setup

For each lab, you get a new GCP project and set of resources for a fixed time at no cost. 

1. Make sure you signed into Qwiklabs using an **incognito window**.
2. Note the lab's access time (for example, 01:30:00) and make sure you can finish in that time block.

There is no pause feature. You can restart if needed, but you have to start at the beginning.

1. When ready, click **Start Lab .** 
2. Note your lab credentials. You will use them to sign in to Cloud Platform Console.![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80684/original/img/eb2f9b843e5c184c.png) 
3. Click **Open Google Console**.
4. Click **Use another account** and copy/paste credentials for **this** lab into the prompts.

If you use other credentials, you'll get errors or **incur charges**.

1. Accept the terms and skip the recovery resource page.

Do not click **End** unless you are finished with the lab or want to restart it. This clears your work and removes the project.

## Start Cloud Shell

#### Activate Google Cloud Shell

From the GCP Console click the Cloud Shell icon on the top right toolbar:

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80684/original/img/55efc1aaa7a4d3ad.png)

Then click "Start Cloud Shell":

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80684/original/img/f517a222b0b2e1c4.png)

It should only take a few moments to provision and connect to the environment:

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/80684/original/img/7ffe5cbb04455448.png)

This virtual machine is loaded with all the development tools you'll  need. It offers a persistent 5GB home directory, and runs on the Google  Cloud, greatly enhancing network performance and authentication.  Much,  if not all, of your work in this lab can be done with simply a browser  or your Google Chromebook.

Once connected to the cloud shell, you should see that you are  already authenticated and that the project is already set to your *PROJECT_ID*.

Run the following command in the cloud shell to confirm that you are authenticated:

```
gcloud auth list
```

**Command output**

```
Credentialed accounts:
 - <myaccount>@<mydomain>.com (active)
```

**Note:** `gcloud` is the powerful and unified command-line tool for Google Cloud Platform. Full documentation is available from [https://cloud.google.com/sdk/gcloud](https://cloud.google.com/sdk/gcloud/). It comes pre-installed on Cloud Shell. You will notice its support for tab-completion.

```
gcloud config list project
```

**Command output**

```
[core]
project = <PROJECT_ID>
```

If it is not, you can set it with this command:

```
gcloud config set project <PROJECT_ID>
```

**Command output**

```
Updated property [core/project].
```

## Copy trained model

### **Step 1**

Set necessary variables and create a bucket

```
REGION=us-central1
BUCKET=$(gcloud config get-value project)
TFVERSION=1.7

gsutil mb -l ${REGION} gs://${BUCKET}
```

### **Step 2**

Copy trained model into your bucket

```
gsutil -m cp -R gs://cloud-training-demos/babyweight/trained_model gs://${BUCKET}/babyweight
```

## Deploy trained model

### **Step 1**

Set necessary variables 

```
MODEL_NAME=babyweight

MODEL_VERSION=ml_on_gcp

MODEL_LOCATION=$(gsutil ls gs://${BUCKET}/babyweight/export/exporter/ | tail -1)
```

### **Step 2**

Deploy trained model

```
gcloud ml-engine models create ${MODEL_NAME} --regions $REGION

gcloud ml-engine versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version $TFVERSION
```

## Deploy App Engine app

### **Step 1**

Clone the course repository

```
cd ~

git clone https://github.com/GoogleCloudPlatform/training-data-analyst
```

### **Step 2**

Complete the TODOs in `/application/main.py` and `/application/templates/form.html`

```
cd training-data-analyst/courses/machine_learning/deepdive/06_structured/serving
```

### **Step 3**

Run the provided deployment script to create and deploy your App Engine app

```
./deploy.sh
```

Note: choose a region for App Engine when prompted

### **Step 3**

Once the deploy.sh script runs successfully and your App Engine app  is deployed, go to the url https://YOUR-PROJECT-ID.apspot.com to start  off with the web form front-end that will make calls through the App  Engine app.

Note: replace YOUR-PROJECT-ID in the url with the project id for your GCP project

©Google, Inc. or its affiliates. All rights reserved. Do not distribute.

# [[eof]]

