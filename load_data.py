


# Load CSV data from files and URL

import csv
import numpy
from urllib.request import urlopen

# loading using csv reader
filename = 'data/pima-indians-diabetes.csv'
in_file = open(filename, 'rt') # rt reads data as ascii
reader = csv.reader(in_file, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = numpy.array(x).astype('float')
print(data.shape) # gives rows, cols
in_file.close()

# load data with numpy directly
in_file = open(filename, 'rt') # rt reads data as ascii
loaded = numpy.loadtxt(in_file, delimiter=',')
print(loaded.shape)
in_file.close()



