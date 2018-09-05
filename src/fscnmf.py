import pandas as pd
import numpy as np
from tqdm import tqdm
from numpy.linalg import inv
from scipy import sparse

class FSCNMF:
    """
    Fused Structure-Content NMF Machine.
    """
    def __init__(self, A, X, args):
        """
        Set up model and weights.
        :param A: Adjacency target matrix. (Sparse)
        :param X: Feature matrix.
        :param args: Arguments object for model.
        """
        self.A = A
        self.X = X
        self.args = args
        self.init_weights()

    def init_weights(self):
        """
        Setup basis and feature matrices.
        """
        self.U = np.random.uniform(0,1,(self.A.shape[0],self.args.dimensions))
        self.V = np.random.uniform(0,1,(self.args.dimensions,self.X.shape[1]))
        self.B_1 = np.random.uniform(0,1,(self.A.shape[0],self.args.dimensions))
        self.B_2 = np.random.uniform(0,1,(self.args.dimensions,self.A.shape[0]))

    def update_B1(self):
        """
        Update node bases.
        """
        simi_term = self.A.dot(np.transpose(self.B_2)) + self.args.alpha_1*self.U
        regul = self.args.alpha_1*np.eye(self.args.dimensions)+self.args.alpha_2*np.eye(self.args.dimensions)
        covar_term = inv(np.dot(self.B_2, np.transpose(self.B_2))+regul)
        self.B_1 = np.dot(simi_term,covar_term)
        self.B_1[self.B_1 < self.args.lower_control] =  self.args.lower_control

    def update_B2(self):
        """
        Update node features.
        """
        covar_term = inv(np.dot(np.transpose(self.B_1),self.B_1)+self.args.alpha_3*np.eye(self.args.dimensions))
        simi_term = self.A.dot(self.B_1).transpose()
        self.B_2 = covar_term.dot(simi_term)
        self.B_2[self.B_2 < self.args.lower_control] =  self.args.lower_control


    def update_U(self):
        """
        Updeate feature basis.
        """
        simi_term = self.X.dot(np.transpose(self.V)) + self.args.beta_1*self.B_1
        regul = self.args.beta_1*np.eye(self.args.dimensions)+self.args.beta_2*np.eye(self.args.dimensions)
        covar_term = inv(np.dot(self.V, np.transpose(self.V))+regul)
        self.U = np.dot(simi_term,covar_term)
        self.U[self.U < self.args.lower_control] =  self.args.lower_control

    def update_V(self):
        """
        Update features.
        """
        covar_term = inv(np.dot(np.transpose(self.U),self.U)+self.args.beta_3*np.eye(self.args.dimensions))
        simi_term = np.dot(np.transpose(self.U),self.X) 
        self.V = np.dot(covar_term,simi_term)
        self.V[self.V < self.args.lower_control] =  self.args.lower_control

    def optimize(self):
        """
        Run power iterations.
        """
        for i in tqdm(range(self.args.iterations)):
            self.update_B1()
            self.update_B2()
            self.update_U()
            self.update_V()

    def save_embedding(self):
        """
        Saving the target matrix.
        """
        print("Saving the embedding.")
        self.target = self.args.gamma*self.B_1+(1-self.args.gamma)*self.U
        self.out = np.concatenate([np.array(range(self.A.shape[0])).reshape(-1,1),self.target],axis=1)
        self.out = pd.DataFrame(self.out,columns = ["id"] + map(lambda x: "X_"+str(x),range(self.args.dimensions)))
        self.out.to_csv(self.args.output_path, index = None)
