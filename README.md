# Image Synthesis using DCGAN

---

## Instructions to use - (Windows)

#### 1. Get the Dataset from the [URL](https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset/discussion) and extract the contents into a folder named "faces" in the project directory
#### 2. Create a folder named "savedstate" in the projecct directory

THE DIRECTORY STRUCTURE SHOULD LOOK LIKE THIS -

```
DCGAN/
|   
|-- faces/
|   |-- man/
|   |-- woman/
|   
|-- savedstate/
|
|-- config.py       
|-- inference.ipynb
|-- model.py
|-- README.md
|-- requirements.txt
|-- scriptwin32.py
|-- train.py
|-- utils.py
```

#### Make sure you have python3 installed, then enter the following commands in an Elevated Terminal to create and activate a python virtual environment
```
python -m venv myenv
cd .\myenv\Scripts\
.\Activate.ps1   
```
##### NOTE - In some systems, User might need to temporarily bypass execution policy
```
Set-ExecutionPolicy Bypass -Scope Process
```
Then run the command below to install all the dependencies in the venv
```
pip install -r requirements.txt
```
This will create a python venv and install all the required dependencies in that venv

#### 3. Install jupyter notbook and iPykernel 
iPykernel is used to register the venv you just created and use it as a kernel in jupyter notebook instance.


Make sure you are in the Project directory and not \myenve\scripts when you run the jupyter notebook command.
```
pip install notebook
pip install ipykernel
python -m ipykernel install --user --name=myenv
jupyter notebook
```

#### 4. Running inference.ipynb
Open the inference.ipynb file in the jupyter notebook instance.

##### NOTE- You need to edit 'config.py' and individual paths to pretrained models in the 'inference.ipynb' to ensure error-free usage.

Select the ipykernel we just created from the kernel dropdown menu.
The kernel should be under "Other kernels" and named "myenv".
Note: You may need to manually install some dependencies which give an error when running inference.ipynb

#### Models
If you wish to use pretrained models use this [URL](https://mega.nz/folder/5qEX2AyR#hThDVv4r1gHgFNM_2uFqCQ) to download the files.

Or you can train a model yourself ig you have enough time and compute resources.

If you wish to use the pretrained models, You just have to extract the .zip into the savedsaved state folder