import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GSE6011 = pd.read_csv("Dataset_GSE6011.csv")
GSE19303 = pd.read_csv("Dataset_GSE19303.csv")
GSE38417 = pd.read_csv("Dataset_GSE38417.csv")
GSE42955 = pd.read_csv("Dataset_GSE42955.csv")
GSE6011 = GSE6011.drop(columns=['UniGene title','UniGene symbol','UniGene ID','Platform_CLONEID','Platform_ORF','Platform_SPOTID'],axis=1)
GSE38417 = GSE38417.drop(columns=['UniGene title','UniGene symbol','UniGene ID','Platform_CLONEID','Platform_ORF','Platform_SPOTID'],axis=1)
GSE19303 = GSE19303.drop(columns=['UniGene title','UniGene symbol','UniGene ID','Platform_CLONEID','Platform_ORF','Platform_SPOTID'],axis=1)
GSE42955 = GSE42955.drop(columns=['UniGene title','UniGene symbol','UniGene ID','Platform_CLONEID','Platform_ORF','Platform_SPOTID'],axis=1)
GSE6011.to_csv("Dataset_GSE6011.csv",index=False)
GSE19303.to_csv("Dataset_GSE19303.csv",index=False)
GSE38417.to_csv("Dataset_GSE38417.csv",index=False)
GSE42955.to_csv("Dataset_GSE42955.csv",index=False)