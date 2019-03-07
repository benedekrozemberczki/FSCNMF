import pandas as pd
import numpy as np
from tqdm import tqdm
from numpy.linalg import inv
from scipy import sparse
from texttable import Texttable

class FSCNMF(object):
    """
    Fused Structure-Content Non-negative Matrix Factorization Machine Abstract Class.
    For details see:  https://arxiv.org/pdf/1804.05313.pdf.
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
        self.losses = []

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
        Update feature basis.
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
        pass

    def calculate_loss(self, iteration):
        """
        Calculate regularization losses at a given iteration.
        :param iteration: Iteration number.
        """
        loss_B_1 = self.args.alpha_2*np.sum(np.square(self.B_1))
        loss_B_2 = self.args.alpha_3*np.sum(np.square(self.B_2))
        loss_U = self.args.beta_2*np.sum(np.square(self.U))
        loss_V = self.args.beta_3*np.sum(np.square(self.V))
        self.losses.append([iteration+1, loss_B_1, loss_B_2, loss_U, loss_V])


    def loss_printer(self):
        """
        Printing the losses in tabular format.
        """
        t = Texttable() 
        t.add_rows([["Losses"]])
        print(t.draw())
        t = Texttable() 
        t.add_rows([["Iteration","Loss B1", "Loss B2", "Loss U", "Loss V"]] +  self.losses)
        print(t.draw())

    def optimize(self):
        """
        Run power iterations.
        """
        for iteration in tqdm(range(self.args.iterations)):
            self.update_B1()
            self.update_B2()
            self.update_U()
            self.update_V()
            if (iteration+1) % 20 == 0:
                self.calculate_loss(iteration)
        self.loss_printer()


    def save_embedding(self):
        """
        Saving the target matrix.
        """
        print("Saving the embedding.")
        self.out = self.args.gamma*self.B_1+(1-self.args.gamma)*self.U
        self.out = np.concatenate([np.array(range(self.A.shape[0])).reshape(-1,1),self.out],axis=1)
        self.out = pd.DataFrame(self.out,columns = ["id"] + [ "X_"+str(dim) for dim in range(self.args.dimensions)])
        self.out.to_csv(self.args.output_path, index = None)


class DenseFSCNMF(FSCNMF):
    """
    Dense Fused Structure-Content Non-negative Matrix Factorization Machine.
    """
    def update_V(self):
        """
        Update features.
        """
        covar_term = inv(np.dot(np.transpose(self.U),self.U)+self.args.beta_3*np.eye(self.args.dimensions))
        simi_term = np.dot(np.transpose(self.U),self.X) 
        self.V = np.dot(covar_term,simi_term)
        self.V[self.V < self.args.lower_control] =  self.args.lower_control
 

class SparseFSCNMF(FSCNMF):
    """
    Sparse Fused Structure-Content Non-negative Matrix Factorization Machine.
    """
    def update_V(self):
        """
        Update features.
        """
        covar_term = inv(np.dot(np.transpose(self.U),self.U)+self.args.beta_3*np.eye(self.args.dimensions))
        simi_term = self.X.transpose().dot(self.U)
        self.V = np.dot(simi_term,covar_term).transpose()
        self.V[self.V < self.args.lower_control] =  self.args.lower_control
