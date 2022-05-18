# Aneurysm-Detection

## Setup

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