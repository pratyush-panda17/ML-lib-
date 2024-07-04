import pandas as pd
class PreProcessor():
    def __init__(self,file):
        self.df = pd.read_csv(file)
    
    def splitDf(self,division):
        if division>0 and division <100:
            division = division /100
            training = self.df.iloc[0:int(division*len(self.df.index))]
            test = self.df.iloc[int(division*len(self.df.index)):]
            return (training,test)
        
    def splitList(ls,division):
        if division>0 and division <100:
            division = division /100
            training = ls[0:int(len(ls)*division)]
            test = ls[int(len(ls)*division):]
            return (training,test)
    
    def getFirstXrows(self,n):
        return self.df.iloc[0:n]

    
    def colToList(df,col_name):
        return df[col_name].tolist()
    




