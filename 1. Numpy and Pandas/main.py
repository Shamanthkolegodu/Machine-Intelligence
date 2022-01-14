#This weeks code focuses on understanding basic functions of pandas and numpy 
#This will help you complete other lab experiments


# Do not change the function definations or the parameters
import numpy as np
import pandas as pd

#input: tuple (x,y)    x,y:int 
def create_numpy_ones_array(shape):
	#return a numpy array with one at all index
	array=np.ones(shape)
	return array

#input: tuple (x,y)    x,y:int 
def create_numpy_zeros_array(shape):
	#return a numpy array with zeros at all index
	array=np.zeros(shape)
	return array

#input: int  
def create_identity_numpy_array(order):
	#return a identity numpy array of the defined order
	if(order>0):
	        array=np.identity(order)
	return array

#input: numpy array
def matrix_cofactor(array):
	#return cofactor matrix of the given array
        def minor(array,i,j):
                sub_array=np.delete(array,i,0)
                sub_array=np.delete(sub_array,j,1)
                determinant_sub_array=np.linalg.det(sub_array)
                return determinant_sub_array

        m=array.shape[0]                
        n=array.shape[1]
        cofactor_matrix=np.zeros((m,n))
        for i in range(m):
                for j in range(n):
                        cofactor_matrix[i][j]=((-1)**(i+j))*minor(array,i,j)
        return cofactor_matrix


#Input: (numpy array, int ,numpy array, int , int , int , int , tuple,tuple)
#tuple (x,y)    x,y:int 
def f1(X1,coef1,X2,coef2,seed1,seed2,seed3,shape1,shape2):
	#note: shape is of the forst (x1,x2)
	#return W1 x (X1 ** coef1) + W2 x (X2 ** coef2) +b
	# where W1 is random matrix of shape shape1 with seed1
	# where W2 is random matrix of shape shape2 with seed2
	# where B is a random matrix of comaptible shape with seed3
	# if dimension mismatch occur return -1
	ans=None
	np.random.seed(seed1)
	W1=np.random.rand(shape1[0],shape1[1])
	np.random.seed(seed2)
	W2=np.random.rand(shape2[0],shape2[1])
	
	if W1.shape[1]==X1.shape[0]:
	        A1=np.matmul(W1,X1**coef1)
	else:
	        return -1
	        
	if W2.shape[1]==X2.shape[0]:
	        A2=np.matmul(W2,X2**coef2)
	else:
	        return -1;
	if A1.shape==A2.shape:
	        np.random.seed(seed3)
	        B=np.random.rand(A1.shape[0],A2.shape[1])
	        ans=A1+A2+B
	else:
	        return -1
	
	return ans



def fill_with_mode(filename, column):
        """
        Fill the missing values(NaN) in a column with the mode of that column
        Args:
        filename: Name of the CSV file.
        column: Name of the column to fill
        Returns:
        df: Pandas DataFrame object.
        (Representing entire data and where 'column' does not contain NaN values)
        (Filled with above mentioned rules)
        """
        df= pd.read_csv(filename)
        df[column] = df[column].fillna(df[column].mode()[0])
        return df

def fill_with_group_average(df, group, column):
        """
        Fill the missing values(NaN) in column with the mean value of the 
        group the row belongs to.
        The rows are grouped based on the values of another column

         Args:
        df: A pandas DataFrame object representing the data.
        group: The column to group the rows with
        column: Name of the column to fill
        Returns:
        df: Pandas DataFrame object.
        (Representing entire data and where 'column' does not contain NaN values)
        (Filled with above mentioned rules)
        """
        #df= pd.read_csv(df)
        x=df.groupby(group)
        df[column].fillna(x[column].transform('mean'),inplace=True)
        return df



def get_rows_greater_than_avg(df, column):
        """
        Return all the rows(with all columns) where the value in a certain 'column'
         is greater than the average value of that column.

        row where row.column > mean(data.column)
        
         Args:
            df: A pandas DataFrame object representing the data.
             column: Name of the column to fill
         Returns:
            df: Pandas DataFrame object.
	"""
        col_mean = df[column].mean()
        df = df.loc[df[column]>col_mean]
        return df

