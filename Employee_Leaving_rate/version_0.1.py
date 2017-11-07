#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 00:51:58 2017

@author: dipankar_nath
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('HR_comma_sep.csv')

#Making the left column as the last column for easier extraction, to be used as "y" in this particular kernel

col_names = ['satisfaction_level',
             'last_evaluation',
             'number_project',
             'average_montly_hours',
             'time_spend_company',
             'Work_accident',
             'salary',
             'promotion_last_5years',
             'sales',
             'left']
dataset = dataset.reindex(columns = col_names)

## Data Preprocessing part

#Randomization
dataset = dataset.sample(frac = 1)

#Encode all categorical data

