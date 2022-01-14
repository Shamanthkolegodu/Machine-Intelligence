import numpy as np


class KNN:
    def __init__(self, k_neigh, weighted=False, p=2):

        self.weighted = weighted
        self.k_neigh = k_neigh
        self.p = p

    def fit(self, data, target):
        self.data = data
        self.target = target.astype(np.int64)

        return self

    def find_distance(self, x):
        N=len(x)
        data=self.data
        p=self.p
        M=len(data)
        matr=np.zeros((N, M))
        for i in range(0,len(x)):
            for j in range(0,len(data)):
                sum=0
                for k in range(0,len(x[0])):
                    sum+=(abs(x[i][k]-data[j][k]))**p
                sum=sum**(1/p)
                matr[i][j]+=sum
        return matr

    def k_neighbours(self, x):
        k=self.k_neigh
        neigh_dists=np.zeros((len(x),k))
        idx_of_neigh=np.zeros((len(x),k))
        matr=self.find_distance(x)
        for i in range(len(matr)):
            dicti={}
            for j in range(len(matr[0])):
                dicti[matr[i][j]]=j;
            dicti = dicti.items()
            dicti=sorted(dicti)
            for j in range(k):
                neigh_dists[i][j]=dicti[j][0]
                idx_of_neigh[i][j]=int(dicti[j][1])
        return([neigh_dists,idx_of_neigh])
    
    
    def predict(self, x):
        pred=[]
        lis=self.k_neighbours(x)
        valu=lis[0];pos=lis[1]        
        target=self.target
        attri=len(x[0])
        if self.weighted:
            for i in range(len(valu)):
                dicti=dict()
                for k in range(attri):
                    dicti[k]=0
                for j in range(len(valu[0])):
                    tar=target[int(pos[i][j])]
                    dicti[tar]+=(valu[i][j]+0.00001)**-1
                a=sorted(dicti.items(),key=lambda kv:(kv[1], kv[0]),reverse=True)
                pred.append(a[0][0])
        else:
            for i in range(len(valu)):
                dicti={}
                max_no=0;attr=-1;
                for k in range(attri):
                    dicti[k]=0
                for j in range(len(valu[0])):
                    tar=target[int(pos[i][j])]
                    dicti[tar]+=1
                list=[(k,v) for k,v in dicti.items()]
                for h in list:
                    if(h[1]>max_no):
                        max_no=h[1]
                        attr=h[0]
                pred.append(attr)
        return(pred)

    def evaluate(self, x, y):
        acc=sum(self.predict(x)==y)/len(y)*100
        return acc