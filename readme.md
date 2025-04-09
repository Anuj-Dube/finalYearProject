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

```git clone <url>```

Step 1:
```
conda create -n animea python=3.10.19\6
pip install -r requirements.txt
```

step 2:
```
conda activate animea
```

step 3:
```
streamlit run main.py
```