import pandas as pd
from collections import Counter
import numpy as np

class TDR():
  def __init__(self,data,addID=True):
    """
        Parameters:
        -----------
        data : pandas.DataFrame or list-like
            Input data to be processed. If not a DataFrame, it will be converted to one.
    """

    self.data = data
    self.addID = addID
    if not isinstance(self.data, pd.DataFrame):
      self.data = pd.DataFrame(self.data)


  def core(self):
    """
    If dataframe have not ID column addID is must True.

    You can select important columns according to their importance level.

    This function is a topological dimensionality reduction algorithm.

    :param converted df, addID=True:
    :return numpy array [index of important column, importance level]:
    """

    if self.addID:
      df.insert(0,"ID",np.arange(1,df.shape[0]+1))

    ##Initialize-0
    ds0=self.data.shape[0]
    ds1=self.data.shape[1]

    df_y=self.data.iloc[:,-1].values
    df_0=self.data.iloc[:,0].values

    IC=[] #important column and importance level
    BaseRla=set()
    BaseB=set()
    X=set()
    cl_lA=[list(range(1, ds1-1))] + [list(range(1, ds1-1))[:i] + list(range(1, ds1-1))[i+1:] for i in range(ds1-2)]
    ##Initialize-0

    ##Shine Examples
    for i in range(ds0):
      if int(df_y[i])==0:
        X.add(df_0[i])
    ##Shine Examples

    for p in range(ds1-1):

      ##Initialize-1
      Rla=[]
      B=[]
      U_R = []
      U = list(df_0)
      cl_l=cl_lA[p]
      print(self.data.columns[p])
      print(cl_l)
      ##Initialize-1

      ##Equivalence Classes
      df_s=self.data.iloc[:,cl_l].values
      while U:
        i = U[0]
        chc = [i]
        for j in U:
            if np.array_equal(df_s[i - 1], df_s[j - 1]) and i != j:
                chc.append(j)
        U = [x for x in U if x not in chc]
        U_R.append(list(chc))
      ##Equivalence Classes
      #print(U_R)

      ##Lower Approximation-Border
      for i in U_R:
        for j in i:
          if set(i).issubset(X):
            Rla.append(j)
          elif not set(i).isdisjoint(X):
            B.append(j)
      ##Lower Approximation-Border


      print("Examples",U)
      print("Shine Examples",X)
      print("Equivalence Classes",U_R)
      print("Lower Approximation",Rla)
      print("Border",B)
      print("*"*75)


      if p>0:
        if not (BaseRla==set(Rla) and BaseB==set(B)):
            IC.append([p, len(BaseB.symmetric_difference(set(B)))])
      else:
        BaseRla=set(Rla)
        BaseB=set(B)

    return np.array(IC)

  def transform(self):
    """
    This function convert data into categories (1,2,3,4,...) based on standard deviation.

    If the convert is not correct and the decimal part of your data is high, you can increase the round_level variable.

    If there is an ID column in the dataset, please cancel this column.

    :param df:
    :return convert df:
    """
    self.data_t = self.data.copy()
    self.transform_dict = {}

    round_level = 3
    for i in range(self.data_t.shape[1]-1):
      # Convert the column to numeric type, coercing errors to NaN
      used_class_list = [] # To collect used classes
      self.data_t.iloc[:, i] = pd.to_numeric(self.data_t.iloc[:, i], errors='coerce') # Convert the i-th column of self.data_t to numeric values, setting non-convertible entries to NaN
      self.transform_dict[i] = {} # Information about which values ​​are assigned to which classes

      c = statistics.stdev(self.data_t.iloc[:,i].dropna())
      d = 1

      while len(used_class_list) != (self.data_t.iloc[:,i].shape[0] - self.data_t.iloc[:, i].isna().sum()):
        try:
          v = round(np.nanmax([t for j,t in enumerate(self.data_t.iloc[:,i]) if (j not in used_class_list)]) - c, round_level) # normal:3

          #print(v,c,(df.iloc[:,i] > v).sum())

          # Converts class
          for j, k in enumerate(self.data_t.iloc[:,i]):

            if (j not in used_class_list) and (k>=v):
              self.data_t.iloc[j,i] = d
              self.transform_dict[i][d] = v
              used_class_list.append(j)

          d += 1
        except Exception as e:
          print("Rise an Error: ",e)
          break
      

 



 
