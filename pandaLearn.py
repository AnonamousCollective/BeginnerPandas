import pandas as pd
from pandas import Series, DataFrame
from pandas_datareader import data as web
import numpy as np
from numpy import nan as NA



#--- SERIES ---
#A Series is a one-dimensional array-like object containing an array
# of data (of any NumPy data type) and an associated array of data labels,
# called its index.
obj = Series([4, 7, -5, 3])
#print(obj) # outputs the array (right) with its resulting index number (left)
obj.values #outputs an array and the data type of obj's values
obj.index #outputs an array and the data type of obj's indexes

#a customised index can also be creted:
obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2

#indexes can be used to select or change the value of the values:
obj2['a'] #outputs -5
obj2['d'] = 6 # changes the array to [6,7,-5,3]
obj2[['c', 'a', 'd']] # here indexes are aoutputted with the value
obj2[obj2 > 0] # selects all values > zero
obj2 * 2
np.exp(obj2) #

#SERIES can be treated like a fixed length ordered DICT
#if you have data in a dict you can create a series from it
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(sdata) # here the index will have the dict keys in sorted order

states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index=states)
#As California is not in sdata its value become NaN (not a number)
#NaN is used by pandas to mark missing or NA values
#The isnull and notnull function should be used to detect missing data.
pd.isnull(obj4) #outputs True or False where relevant
pd.notnull(obj4) #outputs True or False where relevant

#A critical Series feature is that it automatically aligns differently
#indexed data in arithmetic operations
obj3 + obj4 #only California and Utah have NaN values

#Both the Series object and its index have a name attribute
obj4.name = 'population'
obj4.index.name = 'state'
#print(obj4)
#-------------------------------------------------------------------------


#--- DATAFRAME ---
#represents a tabular datadata structure, containing an ordered
#collection of columns
#there are numerous ways to construct a data-frame
data = {'state':['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year':[2000,2001,2002,2001,2002],
        'pop':[1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)
print(frame) #the resulting frame automatically has the index assigned to the rows

DataFrame(data, columns=['year', 'state', 'pop']) #alters the positions of the columns

#if you pass a column that isn't included in 'data' it will show as NA
frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                    index=['one', 'two', 'three', 'four', 'five'])
print(frame2)
#'debt' column will have NA values
# the index has also been changed from traditional list indexing to worded

frame2.columns #outputs the column names
frame2['state'] #a column can be retrieved with dict-like notation
frame2.year #columns can also be outputted like this

#rows can be retrieved by position or name by a couple of methods
frame2.loc['two'] #loc has to be used

#columns can be modified by assignment
frame2['debt'] = 16.5 #assigns 16.5 to all values in the 'debt' column
frame2.debt = np.arange(5.) #assigns float values 0.0-4.0 for the 1st 5 rows
#when assigning lists to DataFrames the list length must exactly match the column length

#if you assign Series it will conform to the DataFrame's index and insert any
#missing values as NA
val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
#series called val created
frame2.debt = val

#assigning a column that doesn't exist will create a new column
#and the 'del' keyword will delete columns just as in a dict
frame2['eastern'] = Series(['oned', 'fourd'], index=['one', 'four'])
#values need to be added when a column is added like this
frame2['eastern'] = frame2.state == 'Ohio' #gives boolean values

del frame2['eastern'] #deletes the eastern column


#another common form of data is a nested dict of dicts format
pop = {'Nevada': {2001:2.4, 2002:2.9},
        'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
        #this way feels a bit more confusing tho
        #the keys in the inner dicts form the index
frame3 = DataFrame(pop)

#you can also transpose the result:
frame3.T #transposes the DataFrame

pdata = {'Ohio': frame3['Ohio'][:-1],
        'Nevada': frame3['Nevada'][:2]}
        #how you output different rows with this form of data

#If a DataFrame’s index and columns have their name attributes set,
# these will also be displayed:
frame3.index.name = 'year'; frame3.columns.name = 'state'
#adds titles to the rows and columns

#the values in a DataFrame can also be outputed as a 2D matrix:
frame3.values #now a 2D matrix
#---------------------------------------------------------------------------


#--- INDEX OBJECTS ---
#pandas’s Index objects are responsible for holding the axis labels and other
# metadata (like the axis name or names).
obj = Series(range(3), index=['a', 'b', 'c'])
index = obj.index
#running print(index) shows that the dtype is an object
#index objects can't be modified by the user once created
#this is important so that index objects can be shared among data structures
index = pd.Index(np.arange(3))
obj2 = Series([1.5, -2.5, 0], index=index) #both obj1 and obj2 have the same index
#------------------------------------------------------------------------------------


#---REINDEXING ---
#means to create a new object with the data conformed to a new index
obj = Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
#calling reindex on the series rearranges the data according to the new index
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e']) # here we have 1 too many index's
#'e' has a value NaN
obj.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0) # now 'e' has value 0

#reindexing can also be used to forward or back fill values
obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3.reindex(range(6), method='ffill') #forward fills the empty space
obj3.reindex(range(6), method='bfill') #back fills the empty space

#with DataFrame, eindex can alter either the (row) index, columns, or both.

frame = DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'c', 'd'],
        columns=['Ohio', 'Texas', 'California'])

frame2 = frame.reindex(['a', 'b', 'c', 'd']) #index 'b' has NaN values
#columns can also be reindexed using the columns keyword
states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states)
#----------------------------------------------------------------------------


#--- DROPPING ENTRIES FROM AN AXIS ---
obj = Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
newObj = obj.drop('c') # index (row) c has been dropped
obj.drop(['d', 'c'])

#with a DataFrame, index values can be deleted from either axis
data = DataFrame(np.arange(16).reshape((4, 4)),
        index=['Ohio', 'Colorado', 'Utah', 'New York'],
        columns=['one', 'two', 'three', 'four'])

data.drop(['Colorado', 'Ohio']) #drops oth of those indexes
data.drop('two', axis=1) #when dropping columns we have to specify its axis=1
#-----------------------------------------------------------------------------------


#--- INDEXING, SELECTION, AND FILTERING ---
obj = Series(np.arange(4.), index=['a', 'b', 'c', 'd'])

obj['b'] #and
obj[1] #output the same value

obj[2:4] #,
obj[['c', 'd']] #,
obj[obj>1] #,
obj[[2, 3]] #and
obj['c':'d'] #output the same value
#as you can see slicing with labels behaves differently that normal py slicing
#as the endpoint is inclusive

data = DataFrame(np.arange(16).reshape((4, 4)),
        index=['Ohio', 'Colorado', 'Utah', 'New York'],
        columns=['one', 'two', 'three', 'four'])
#we do not need to specify the axis this time round
data['two'] #selects the column labelled two
data[['three', 'one']] #outputs column three and one

data[:2] #outputs the 1st two rows: 0 & 1
data[data['three']>5] #outputs values of the table only where col threes values > 5

data < 5 #outputs the DataFrame with boolean values

data[data < 5] = 0 #any value < 5 is becomes 0

data.loc['Colorado', ['two', 'three']] #outputs data at row: colorado, col:two & three
data.loc[['Colorado', 'Utah'], ['four', 'one', 'two']] #im sure u can figure what this does
data.iloc[2,] #outputs the values in the 3rd row
data.loc[:'Utah', 'two'] #all values in rows up to Utah and col two

data[data['three']>5][:2]
#-----------------------------------------------------------------------------------


#--- ARITHMETIC AND DATA ALIGNMENT ---
# When adding together objects, if any index pairs are not
#the same, the respective index in the result will be the
#union of the index pairs.

s1 = Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])

s1 + s2 # organises the index in order
#indexes that dont feature across both series have NaN values

#for DataFrame's, alignment is performed on both the rows and columns
df1 = DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'),
        index=['Ohio', 'Texas', 'Colorado'])

df2 = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
        index=['Utah', 'Ohio', 'Texas', 'Oregon'])

df1 + df2 #outputs similar answers as s1 + s2

#wen working with differently indexed objects, you may want to fill
#NaN with 0.0
df1 = DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))
df2 = DataFrame(np.arange(20.).reshape((4, 5)), columns=list('abcde'))
df1.add(df2, fill_value=0.0) #all NaN values in df1 converted to 0.0
#                              so df2's values can be successfully added

#when reindexing  a DataFrame or Series,
#you can also specify a different fill value
df1.reindex(columns=df2.columns, fill_value=0) #df1 now has df2's columns

#operations between DataFrame and Series:
arr = np.arange(12.).reshape((3, 4))
arr[0] #outputs the 1st row of arr

arr - arr[0] #the 1st row is now filled with 0.0

frame = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
        index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.iloc[0]

#arithmetic functions between DataFrames and Series match the index
frame - series #is the same as arr - arr[0]

#if an index is not found in either the DataFrame or Series
#objects will be re-indexed to form the union
series2 = Series(range(3), index=['b', 'e', 'f'])
frame + series2

#cols in a DataFrame can be assigned to a Series
series3 = frame['d']
#series3 can now be substracted from eah col in the DataFrame
frame.sub(series3, axis=0)
#---------------------------------------------------------------------------------


#--- FUNCTION APPLICATION AND MAPPING ---
#Numpys ufunc(element-wise array methods) work fine with pandas obj's
frame = DataFrame(np.random.randn(4, 3), columns=list('bde'),
        index=['Utah', 'Ohio', 'Texas', 'Oregon'])
np.abs(frame) #outputs a DataFrame with only positive/absolute values

f = lambda x: x.max() - x.min()
#lambda is a way to define anonymous functions without having to create a
#definition in py

frame.apply(f) #this shows the spread of the values in each col
frame.apply(f, axis=1) #the spread across each row

def f(x):
        return Series([x.min(), x.max()], index=['min', 'max'])
        #index labelling has to be done carefully
frame.apply(f)

format = lambda x: '%.2f' % x 
#outputs values to the 1st TWO digits after the point
frame.applymap(format)
#difference between APPLY  and APPLYMAP:
#apply: takes the whole col as a parameter and assigns result to this
#applymap: take the separate cell value as a parameter and assigns to that
#---------------------------------------------------------------------------------


#--- SORTING AND RANKING ---
obj = Series(range(4), index=['d', 'a', 'b', 'c'])
obj.sort_index() #sorts the Series's row lexicographically

frame = DataFrame(np.arange(8).reshape((2, 4)), index=['three', 'one'],
        columns=['d', 'a', 'b', 'c'])
frame.sort_index(axis=1) #sorts the DataFrame's col lexicographically
frame.sort_index(axis=1, ascending=False) #sorts col in descending order

obj = Series([4, 7, -3, 2, np.nan])
obj.sort_values() #sorts the values in a Series in ascending order
#missing values are sorted to the end of the Series by default

#to sort DataFrames by 1 or more columns, we use 'by'
frame = DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame.sort_values(by='b')
frame.sort_values(by=['a', 'b']) #sorts multiple columns

obj = Series([7, -5, 7, 4, 2, 0, 4])
obj.rank() #outputs the values in the series positional rankings
obj.rank(method='first') #if 2 values are the same they dont get the same rank
#                       instead the value that appears 1st gets the smaller
#                       positional rank.
obj.rank(ascending=False, method='max') #equal values get the same rank - print and see

frame = DataFrame({'b': [4.3, 7, -3, 2], 'a': [0, 1, 0, 1],
        'c': [ -2, 5, 8, -2.5]})
frame.rank(axis=1) #ranks the rows
#---------------------------------------------------------------------------------


#--- AXIS INDEXES WITH DUPLICATE VALUES ---
obj = Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
obj.index.is_unique #outputs 'True' or 'False'

obj['a'] #outputs both 0 and 1

#the same logic extends to DaaFrames
df = DataFrame(np.random.randn(4, 3), index=['a', 'a', 'b', 'b'])
df.loc['b']
#------------------------------------------------------------------------------


#--- SUMMARISING AND COMPUTING DESCRIPTIVE STATISTICS ---
#methods that extract a single value(like the sum or mean)
#are all built to exclude missing data
df = DataFrame([[1.4, np.nan], [7.1, -4.5],
        [np.nan, np.nan], [0.75, -1.3]],
        index=['a', 'b', 'c', 'd'],
        columns=['one', 'two'])

df.sum() #sums the columns
df.sum(axis=1) #sums the rows

#to disable the NA values being disabled use:
df.mean(axis=1, skipna=False)

df.idxmax() #outputs the id which has the max value
df.cumsum(axis=1) #cumsum of the rows

df.describe() #outputs multiple summary statistics in one shot
obj = Series(['a', 'a', 'b', 'c'] * 4)
obj.describe() #outputs a different set of statistics for non-numeric data
#--------------------------------------------------------------------------


''' confused AF!
#--- CORRELATION AND COVARIANCE ---
#considering some DataFrames of stock prices and volumes obtained from
#Yahoo! Finance:
all_data = {}
for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOG']:
        all_data[ticker] = web.get_data_yahoo(ticker, '1/1/2000', '1/1/2010')

price = DataFrame({tic: data['Adj Close']
        for tic, data in all_data.items()})

volume = DataFrame({tic: data['Volume']
        for tic, data in all_data.items()})

returns= price.pct_change()
returns.tail() #tails() means outputs the last 5 rows

#the 'corr' method of Series computes the correlation of the overlapping,
#non-NA, aligned-by-index values in two Series. Relatedly, 'cov'
#computes the covariance:
returns.MSFT.corr(returns.IBM)
returns.MSFT.cov(returns.IBM)

#DataFrame’s corr and cov methods, on the other hand, return a full
#correlation or covariance matrix as a DataFrame, respectively:
returns.corr()
returns.cov()

#Using DataFrame’s corrwith method, you can compute pairwise
#correlations between a DataFrame’s columns or rows with another
#Series or DataFrame.
returns.corrwith(returns.IBM)

#Passing a DataFrame computes the correlations of matching column names.
returns.corrwith(volume)
#----------------------------------------------------------------------------
'''


#--- UNIQUE VALUES, VALUE COUNTS, AND MEMBERSHIP ---
obj = Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
uniques = obj.unique() #identifies the unique values
uniques.sort() #sorts the output of the unique values

obj.value_counts() #states the unsorted frequency of the unique values
pd.value_counts(obj.values, sort=False) #indexes are in order but the values are not

mask = obj.isin(['b', 'c'])
mask #outputs Boolean values
#-------------------------------------------------------------------------------------


#--- HANDLING MISSING DATA ---
#NaN is used to represent missing data
string_data = Series(['aardvark', 'artichoke', np.nan, 'avocado'])
string_data.isnull() #outputs boolean
string_data[0] = None #None also treated as a NaN
string_data.isnull() 

data = Series([1, NA, 3.5, NA, 7])
data.dropna() #returns the series with only non-null data and indexes
data[data.notnull()] #the same as:... data.dropna()

data = DataFrame([[1., 6.5, 3.], [1., NA, NA],
        [NA, NA, NA], [NA, 6.5, 3.]])
data.dropna() #drops any rows containing NaN values
data.dropna(how='all') #only drops rows that are all NaN
data.dropna(how='all', axis=1)

#thresh can be used to keep rows with a certain number of observations
df = DataFrame(np.random.randn(7, 3))
df.loc[:4, 1] = NA; df.loc[:2, 2] = NA #choosing what values are NA
df.dropna(thresh=2) #drops rows with less than 2 non-NaN values
#------------------------------------------------------------------------------


#--- FILLING IN MISSING DATA ---
#instead of potentially discarding any useful data it may be better to fill NaN values
df.fillna(0) #NaN now outputted as 0

#calling fillna with a dict can use a different fill value for each col
df.fillna({1: 0.5, 3: -1}) #fills the 2nd and 4th columns
#fillna returns a new obj, but the old obj can be modified
df.fillna(0, inplace=True) #df has now been modified

#the same interpolation method from re-indexing (LINE 136-158) can be used:
df.loc[2:, 1] = NA; df.loc[4:, 2] = NA
df.fillna(method='ffill', limit=2) #forward-fills only twice in each col

#fillna can even be used to pas the mean or median to other values
data = Series([1., NA, 3.5, NA, 7])
data.fillna(data.mean())
#----------------------------------------------------------------------------------


#--- HIERARCHICAL INDEXING ---
#enables you to have 2+ index levels on an axis

data = Series(np.random.randn(10),
        index=[['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd'],
        [1, 2, 3, 1, 2, 3, 1, 2, 2, 3]])
data.index #outputs the indexes
data['b':'c']
data.loc[['b', 'd']]
print(data)

#Hierarchical indexing plays a critical role in reshaping data and
#group-based operationslike forming a pivot table.
data.unstack() #re-arranges the series into a DataFrame
data.unstack().stack() #reverts the DataFrame into a Series

#With a DataFrame either axis can have a Hierarchical index:
frame = DataFrame(np.arange(12).reshape((4, 3)),
        index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
        columns=[['Ohio', 'Ohio', 'Colorado'],
        ['Green', 'Red', 'Green']])
#The hierarchical levels can have names (as strings or any Python objects).
#If so, these will show up in the console output (don’t confuse the index
#names with the axis labels!):

frame.index.names = ['key1', 'key2']
frame.columns.names = ['state', 'color']
frame['Ohio'] #DataFrame only displays values linked to Ohio
'''
#the cols from the DataFrame can be created like this
MultiIndex.from_arrays([['Ohio', 'Ohio', 'Colorado'],['Green', 'Red', 'Green']],
        names=['state', 'color'])
'''
#-----------------------------------------------------------------------------


#--- REORDERING AND SORTING LEVELS ---
frame.swaplevel('key1', 'key2') #swaps positions of theses indexes
frame.sort_index(0) #sorts the row indexes in lexicographical order
frame.sort_index(1) #sorts the cols indexes
frame.swaplevel(0,1).sort_index(0) #swapped the indexes of the rows and sorted

frame.sum(level='key2') #sums the level of key2
frame.sum(level='color', axis=1) # sums the level color in the col

frame = DataFrame({'a': range(7), 'b': range(7, 0, -1),
        'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'],
        'd': [0, 1, 2, 0, 1, 2, 3]})

frame2 = frame.set_index(['c', 'd']) #new row indexes are c & d, which have now been dropped from the col
frame.set_index(['c', 'd'], drop=False) #now c & d aren't dropped
frame2.reset_index() #the indexes are now reset
#-----------------------------------------------------------------------------------------------


#--- OTHER PANDAS TOPICS ---
'''
ser = Series(np.arange(3.))
ser[-1]
'''#gives an error as pandas don't behave like lists
#but with non-integer indexes it works as there is no room for mistaken meaning with the integers
ser2 = Series(np.arange(3.), index=['a', 'b', 'c'])
ser2[-1]
#---------------------------------------------------------------------------------

#------------------------------------------------------------------------