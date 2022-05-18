# Aneurysm-Detection

## Setup

### File structure
Add a ***data*** folder into your locally cloned repository so that the directory structure looks similar to this:
```
ANEURYSM-DETECTION
│   README.md
│   pyproject.toml
│   ...
│
└───data 
│   │
│   └───training
│   │   │   ...
│   │
│   └───3DUnet_training
│       │   ...
│   
└───externals
│   │
│   └───pytorch3dunet
│       │   ...
│   
└───src
    │   ...
```
The 3DUnet_training data for testing the external repository can be downloaded fom [here](https://osf.io/9x3g2/) (Cell boundary predictions for lightsheet images of Arabidopsis thaliana lateral root).
Our aneurysm dataset was privately shared.
### Poetry shell (virtual env)
#### Install poetry
Follow the [poetry installation guide](https://python-poetry.org/docs/) to install poetry if not already installed.
The following should be sufficient:
```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```
Verify by running
```bash
poetry --version
```
#### Setup env
In repository run following command to setup evnironment:
```bash
poetry install
```
To activate poetry shell run:
```bash
poetry shell
```

### External repositories
To include the external repositories execute the following command:
```bash 
git submodule update --init --recursive 
```
###  VSCode
To include the path for each external library (_example path given_) include this into your `settings.json`:
```
"python.analysis.extraPaths": [
        "/home/user/Aneurysm-Detection/externals/pytorch3dunet"
    ]
```