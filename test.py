from data_wrangling import titanic_train
import re
import numpy as np

#extract the title
def get_title(name):
	pattern = re.compile('([A-Za-z]+)\.')
	match = re.search(pattern, name)
	if match: return match.group(1) 
	return ''

#apply extract function to the dataset
titles = titanic_train['Name'].apply(get_title)
utitles = list(np.unique(titles))
title_mapping = {k: i+1 for k, i in zip(utitles, range(len(utitles)))}

import pdb;pdb.set_trace()

print title_mapping.sort