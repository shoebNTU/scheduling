from typing import OrderedDict
# from numpy.core.fromnumeric import size
import pandas as pd
import streamlit as st 
import numpy as np  
from scipy.spatial import distance
from itertools import combinations
# from scipy.stats import rankdata
# from skcriteria import Data
# from skcriteria.madm.simus import SIMUS
# from skcriteria.madm import closeness

# function to load input
@st.cache(show_spinner=False,suppress_st_warning=True)
def load_input():    
    df_input = pd.read_excel('input.xlsx')    
    return df_input

# function to display cost, risk and fpmk side-by-side for different interventions


# MOO function
def get_grid(h,m):
    combs = np.array([x for x in combinations(range(h+m-1),m-1)]) - np.array(range(m-1))
    grid = np.concatenate((combs[:,0].reshape(-1,1),np.diff(combs),h-combs[:,-1].reshape(-1,1)),axis=1)/h
    return grid

def is_pareto_efficient(points):
    is_efficient = np.arange(points.shape[0])
    n_points = points.shape[0]
    next_point_index = 0
    for i in range(n_points):
        if next_point_index >= points.shape[0]:
            break
        nondominated_point_mask = np.any(points < points[next_point_index],axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]
        points = points[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index+1])
    return is_efficient

def MOO(df):
    inv = df.values
    points_dict = {1:1, 2:500, 3:100, 4:50, 5:25, 6:18, 7:15, 8:12, 9:10, 10:9}
    num_interventions = inv.shape[0]
    num_points = points_dict[num_interventions] if num_interventions <= 10 else 8

    points = get_grid(num_points,num_interventions)
    points = points.dot(inv)
    mask = is_pareto_efficient(points)

    pareto = points[mask]/inv.sum(axis=0)
    inv = inv/inv.sum(axis=0)

    
    dist = []

    for i in range(num_interventions):
        dist.append(np.mean(np.min(distance.cdist((inv)[i].reshape(1,3),pareto),axis=1))*1000+inv.sum(axis=1)[i])
    dist = np.array(dist)
    rank_MOO = dist.argsort().argsort()+1
    # st.text()
    return rank_MOO


# AHP function
class AHP():
    RI = {1:0, 2:0, 3:0.58, 4:0.9, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45}
    consistency = False
    priority_vec = None
    compete = False
    normal = False
    sublayer = None

    def __init__(self,name,size):
        self.name = name
        self.size = size
        self.matrix = np.zeros([size,size])
        self.criteria = [None]*size

    def update_matrix(self,mat,automated=True):
        self.original_matrix = mat
        if not((mat.shape[0] == mat.shape[1]) and (mat.ndim == 2)):
            raise Exception('Input matrix must be squared')
        
        if self.size != len(self.criteria):
            self.criteria = [None]*self.size
        self.matrix = mat
        self.size = mat.shape[0]
        self.consistency = False
        self.normal = False
        self.priority_vec = None
        if automated:
            self.rank()
    
    def input_priority_vec(self,vec):
        if not(vec.shape[1]==1) and (vec.shape[0]==self.size) and (vec.ndim==2):
            raise Exception('Size of input priority vector is not compatible.')
        self.priority_vec = vec
        self.output = self.priority_vec/self.priority_vec.sum()
        self.consistency = True
        self.normal = True

    def rename(self,name):
        self.name = name

    def update_criteria(self,criteria):
        if len(criteria) == self.size:
            self.criteria = criteria
        else:
            raise Exception('Input does not match number of criteria')
    
    def add_layer(self,alternative):
        if not self.criteria:
            raise Exception('Please input criteria before adding new layer')
        self.compete = False
        self.sublayer = OrderedDict()
        self.alternative = alternative
        for i in range(self.size):
            self.sublayer[self.criteria[i]] = AHP(self.criteria[i],len(alternative))
            self.sublayer[self.criteria[i]].update_criteria(self.alternative)

    def normalize(self):
        if self.normal:
            pass
        self.col_sum = self.matrix.sum(axis=0)
        try:
            self.matrix = self.matrix/self.col_sum
        except:
            raise Exception('Error when normalizing on columns')
        else:
            self.normal = True
            self.priority_vec = self.matrix.sum(axis=1).reshape(-1,1)

    def rank(self):
        if self.consistency:
            df = pd.DataFrame(data = self.output, index = self.criteria, columns=[self.name])
            return df
        
        if not self.normal:
            self.normalize()
        
        Ax = self.matrix.dot(self.priority_vec)
        eigen_val = (Ax/self.priority_vec).mean()
        eigen_val = np.linalg.eig(self.original_matrix)[0].max()
        CI = (eigen_val - self.size)/(self.size-1)

        if self.size > 2:
            CR = CI/self.RI[self.size] 
        else:
            CR = 0.0
                   
        if CR<0.1:
            self.consistency = True
            self.output = self.priority_vec/self.priority_vec.sum()
            self.df_out = pd.DataFrame(data = self.output, index = self.criteria, columns = [self.name])
            return self.df_out
        else:
            raise Exception('Consistency is not sufficient to reach a decision')

    def make_decision(self):
        if not self.consistency:
            self.rank()
        if not self.compete:
            temp = True
            arrays = []
            for item in self.sublayer.values():
                item.rank()
                temp = temp and item.consistency
                if temp:
                    arrays.append(item.output)
                else:
                    raise Exception('Please check AHP for {}'.format(item.name))
            
            if temp:
                self.compete = True
            else:
                pass
            self.recommendation = np.concatenate(arrays, axis=1).dot(self.output)
        self.df_decision = pd.DataFrame(data=self.recommendation, index=self.alternative, columns = ['AHP Score'])
        self.df_decision.index.name = 'Alternative'
        self.df_decision['rank'] = self.df_decision['AHP Score'].rank(ascending=False) # AHP-maximize
        return self.df_decision

def AHP_rank(df,criteria_matrix):
    method = AHP('Relative Importance',df.shape[1]) # the second argument is number of objectives
    method.update_criteria(list(df.columns))
    method.update_matrix(criteria_matrix)
    method.add_layer([i for i in range(df.shape[0])]) #adding interventions

    # iterate over )column names --> method.sublayer['col_name'].input_priority_vec(insert corresponding col here)
    # finally method.make_decision
    for col in df.columns: #adding priority vectors associated with objectives corresponding to different interventions
        method.sublayer[col].input_priority_vec(np.array(df[col]).reshape(-1,1))

    return method.make_decision() #['rank']#.values#.astype(int)

class dt_intv():
    
    def __init__(self,time):
        self.year = int(time[0])
        self.month = int(time[1])
        self.time = (self.year-2020)*12 + self.month
    
    def __add__(self,number):
        self.year += (self.month+number-1)//12
        self.month = (self.month+number-1)%12 + 1
        self.time += number
        return self
    
    def __sub__(self,other):
        if type(other) == type(dt_intv([2020,1])):
            return self.time - other.time
        else:
            self.year += (self.month-other-1)//12
            self.month = (self.month-other-1)%12 + 1
            self.time -= other
            return self
            
    def __ge__(self,other):
        return self.time >= other.time
    
    def __lt__(self,other):
        return self.time < other.time
    
    def year_month(self):
        return str(int(self.year))+("0"+str(int(self.month)))[-2:]
            

class Intervention():
    vect = False
    cost = None
    risk = None
    fpmk = None
    
    def __init__(self,d,e,intv_id,fac,num,eff=0.5):
        self.intv_id = intv_id
        self.fac = fac
        self.duration = d
        self.enddate = dt_intv(e)
        self.startdate = dt_intv(e)-d+1
        self.shorttime = int(d*eff)
        self.shortened = 0
        self.original = [d,dt_intv(e),(dt_intv(e)-d+1),num]
        self.rank = 0
        self.num = num
        
    def __contains__(self,time):
        return self.startdate.time <= time and self.enddate.time >= time
    
    def show_original(self):
        return self.original
    
    def add_rank(self,r):
        self.rank = r
        
    def change(self):
        if self.shortened < self.shorttime:
            self.startdate += 1
            self.duration -= 1
            self.shortened += 1
            self.enddate = (dt_intv([self.startdate.year_month()[:4],self.startdate.year_month()[4:]]) + self.duration - 1)
            
        else:
            self.startdate += 1
            self.duration = self.original[0]
            self.enddate  = (dt_intv([self.startdate.year_month()[:4],self.startdate.year_month()[4:]]) + self.duration - 1)
    
    def update_info(self,cost,risk,fpmk):
        self.vect = True
        self.cost = cost
        self.risk = risk
        self.fpmk = fpmk
            
        

