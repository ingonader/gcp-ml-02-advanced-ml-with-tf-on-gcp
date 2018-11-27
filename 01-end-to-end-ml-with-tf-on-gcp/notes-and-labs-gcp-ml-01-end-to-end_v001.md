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

# [[eof]]

