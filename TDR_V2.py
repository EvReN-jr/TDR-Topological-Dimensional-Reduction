import statistics
import numpy as np

class TDR:
  def topological_dimensionality_reduction(self,df,addID=True):
    return self.core(self.transform(df),addID)

  def transform(self,df,round_level=3):
    """
    This function convert data into categories (1,2,3,4,...) based on standard deviation.

    If the convert is not correct and the decimal part of your data is high, you can increase the round_level variable.

    If there is an ID column in the dataset, please cancel this column.

    :param df:
    :return convert df:
    """
    for i in range(df.shape[1]-1):
      c=statistics.stdev(df.iloc[:,i])
      d="1"
      while True:
        try:
          v=round(np.max([t for t in df.iloc[:,i] if type(t)!=str])-c,round_level)# normal:3
          for j,k in enumerate(df.iloc[:,i]):
            if type(k)!=str and k>=v:
              df.iloc[j,i]=d
          d=str(int(d)+1)
        except:
          break
      df.iloc[:,i]=[int(k) for t,k in enumerate(df.iloc[:,i]) if type(k)==str]
      #print(df.iloc[:,i])
    return df

  def core(self,df,addID=True):
    """
    If dataframe have not ID column addID is must True.

    You can select important columns according to their importance level.

    This function is a topological dimensionality reduction algorithm.

    :param df, addID=True:
    :return numpy array [index of important column, importance level]:
    """

    if addID:
      df.insert(0,"ID",np.arange(1,df.shape[0]+1))

    ##Initialize-0
    ds0=df.shape[0]
    ds1=df.shape[1]

    df_y=df.iloc[:,-1].values
    df_0=df.iloc[:,0].values

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
      print(df.columns[p])
      print(cl_l)
      ##Initialize-1

      ##Equivalence Classes
      df_s=df.iloc[:,cl_l].values
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
            IC.append([p,len(BaseB.symmetric_difference(set(B)))])
      else:
        BaseRla=set(Rla)
        BaseB=set(B)

    return np.array(IC)
