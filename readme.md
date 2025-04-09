# Animea Predictions
## Instructions:
### Prerequisites
Download Miniconda by following these instructions for Windows:

Run these commands on cmd.exe
```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o .\miniconda.exe
start /wait "" .\miniconda.exe /S
del .\miniconda.exe
```

Clone this github repository

```git clone https://github.com/Anuj-Dube/finalYearProject.git```

Setup the enviornment and run the application:
```
conda create -n animea python=3.10
conda activate animea
cd finalYearProject
pip install -r requirements.txt
streamlit run main.py
```
