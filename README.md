# mlops-project
## LIPSOL , AN URDU LIP READING APPLICATION
## Install Dependancies
1. python -m venv /env
2. source env/scripts/activate
3. pip install -r requirements.txt

## python version 3.9.0
## Goals:


The goal of this project is to help hearing impaired people to communicate with each other by means of lip reading. Though the project is experimental and is in the initial stages of blooming, our work gives a proof of concept . We have created the first lip reading application in urdu by collecting a small dataset and applying various state of the art english models by tweaking and in some cases re-training them

## Achievements:

hello
We have created two successful models that give considerable wer level and character level accuracy. One is a word level model, the other is a character level model. Though there is still a long way to go when it comes to achieving a good accuracy on unseen speakers, yet the model works well on seen speakers.


## Project Design:


Our models and dataset are pretty substantial for which we decided to apply mlops whist the development of our project. Our project design includes
1.Training model through mlflow to see which parameters best align to give the highest word level and character level accuracy.
2.Being able to maintain the various versions of our model by pushing the preprocessed dataset, dataset and weights to a remote aws s3 bucket using DVC.
3.Being able to retrieve weights and containerize our application so that we can easily run our models on various machines and can quickly exchange with group members which will enable us to see rapid results. We have also containerized the application so that we can eventually make the container public for testing and exploring purposes since this is an R&D project.
4.To visualize rapid development of our flask backend and making an endpoint accessible for testing from all members, we have created a CI/CD of the development branch through jenkins and docker. The jenkins is configured to be triggered whenever there are changes in the repository and exposes port 5000 of ec2 to deploy the containerized applications. Jenkins server is itself working on port 8080 of that machine
5.We have also used airflow dags to parallelize and streamline our tasks. Training different models and then choosing the best model using accuracy thresholding are some of the tasks that are used to choose a trained model/weights that can be deployed or has a certain accuracy that's above the previously deployed model. These training pipelines are scheduled daily. 
6.To ensure that no security breaches occur and code quality is consistent we have added github workflows and branch protection to our central repository which requires us to review each others code before pushing to the development branch

