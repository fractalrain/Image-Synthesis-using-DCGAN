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

#### Make sure you have python3 installed and then run the "scriptwin32.py" file with cmd in the project directory
```
python scriptwin32.py
```
This will create a pythno venv and install all the required dependencies in that venv

