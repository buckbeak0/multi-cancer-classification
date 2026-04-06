# brain-tumor-mri-classification

## How to run this
- Create a project folder named 'Multi cancer Classification'
- Go to the project directory from terminal or open the folder in vscode.
- Clone this repo to your pc
- Use command ```git clone 'https://github.com/buckbeak0/brain-tumor-mri-classification.git'```
- Create a 'models' folder
- Download models and data (id required) from google drive
- create a virtual environment
- Run this on terminal (cmd) ```py -m venv .venv```
- Activate the virtual environment
- ```.venv/Scripts/activate```
- Download the libraries listed in requirements.txt
- Use this command ```pip install -r requirements.txt```
- After successfull completion of every step
- To start the server / run this project
- Use command ```uvicorn main:app --reload```
- A link would pop in the output click on it

## Precautions
- Before starting always make sure venv is activated
- Do not train any model on local pc use online platform like colab
- Close the server using Ctrl + c after usage


### Note - We can change the model in main.py line 19 by just changing the model file name

