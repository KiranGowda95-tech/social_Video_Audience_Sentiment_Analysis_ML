import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

import yaml
import logging

#Logging configuration

logger =logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler=logging.StreamHandler()
console_handler.selLevel(logging.DEBUG)

file_handler=logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s -%(message)s ')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path:str)->dict:
    try:
        pass
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)