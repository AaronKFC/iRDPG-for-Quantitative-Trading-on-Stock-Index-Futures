# RL-Financial
## Requirement

- Python 3.6
- Pytorch 1.4.0

## Install & Run

1. (Optional) `virtualenv env -p python3` & `source env/bin/activate`
2. `pip3 install -r requirements.txt`
   - May have some errors while installing graphic related package on different system
3. `python3 run.py`

## Project Structure

- data
  - **Empty directory**, synchronized data will be saved at here
- origin_data
  - Original source of the dataset
- report
  - **Empty directory**, generated report will be here
- src
  - Source code

## Module Description

- [setting.py](src/setting.py)
  -  Contains all the framework variables ( e.g. dataset path, selected dataset, selected methods, selected feature selection methods )
- [run.py](src/run.py)
  - Entry module that trigger everyother modules to start the process


## Default dataset link

- [AEEEM](http://bug.inf.usi.ch/index.php)


## Reference

