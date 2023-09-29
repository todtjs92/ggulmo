import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader

class FM(nn.Module):
    def __init__(self, df_train_feat , df_train_label , df_valid_feat , df_valid_label , field_dims , embed_dim ,
                 num_epochs , early_stop_trial, learning_rate, reg_lambda, batch_size, device ):

        super().__init__()

        self.df_train_feat = df_train_feat
        self.df_train_label = df_train_label
        self.df_valid_feat = df_valid_feat
        self.df_valid_label = df_valid_label
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.num_epochs = num_epochs
        self.early_stop_trial = early_stop_trial
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.batch_size = batch_size
        self.device = device

        self.build_graph()

    def build_graph(self):

        # 1 linear
        self.linear = FeaturesLinear(self.field_dims)

        # 2 embed for interact
        self.embedding = FeaturesEmbedding(self.field_dims, self.embed_dim)
        self.fm = FactorizationMachine()

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.reg_lambda)

        self.to(self.device)


    def forward(self,x):

        linear_result  =  self.linear(x)
        selected_embedding = self.embedding(x)
        interaction_result = self.fm(selected_embedding)

        result_sum = linear_result + interaction_result

        output = torch.sigmoid(result_sum.squeeze(1))


        return output


    def fit(self):
        train_loader = DataLoader(range(self.df_train_feat.shape[0]), batch_size=self.batch_size, shuffle=True)
        best_AUC = 0
        num_trials = 0

        for epoch in range(1, self.num_epochs + 1):
            # train
            self.train()

            for b, batch_idxes in enumerate(train_loader):

                batch_data = self.df_train_feat[batch_idxes]

                batch_labels = self.df_train_label[batch_idxes]

                loss = self.train_model_per_batch(batch_data, batch_labels)

            # 이거 나중에 체크 , loader 그대로 안쓰고 배치로 맞꿔서 넣어줌.
            self.eval()
            pred_array = self.predict(self.df_valid_feat)
            AUC = roc_auc_score(self.df_valid_label, pred_array)
            logloss = log_loss(self.df_valid_label, pred_array)

            if AUC > best_AUC:
                best_AUC = AUC
                torch.save(self.state_dict(), f"./{self.__class__.__name__}_best_model.pt")
                num_trials = 0

            else:
                num_trials += 1

            if num_trials >= self.early_stop_trial and self.early_stop_trial > 0:
                print(f'Early stop at epoch:{epoch}')
                self.restore()
                break

            print(f'epoch {epoch} train_loss = {loss:.4f} valid_AUC = {AUC:.4f} valid_log_loss = {logloss:.4f}')

        return


    def train_model_per_batch(self, batch_data, batch_labels):

        batch_data = torch.FloatTensor(batch_data).to(self.device)

        batch_labels = torch.FloatTensor(batch_labels).to(self.device)

        self.optimizer.zero_grad()

        logits = self.forward(batch_data)

        loss = self.criterion(logits, batch_labels)
        loss.backward()

        self.optimizer.step()

        return loss

    def predict(self, pred_data):
        # return numpy array

        self.eval()

        pred_data_loader = DataLoader(range(pred_data.shape[0]), batch_size=self.batch_size, shuffle=False)
        pred_array = np.zeros(pred_data.shape[0])
        for b, batch_idxes in enumerate(pred_data_loader):
            batch_data = torch.tensor(pred_data[batch_idxes], dtype=torch.float, device=self.device)

            with torch.no_grad():
                pred_array[batch_idxes] = self.forward(batch_data).cpu().numpy()

        print(pred_array[batch_idxes])


        return pred_array

    def restore(self):
        with open(f"./{self.__class__.__name__}_best_model.pt", 'rb') as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)

class FeaturesLinear(nn.Module):
    def __init__(self, field_dims , output_dim = 1):
        super().__init__()

        # embedding for linear multiply
        self.fc = torch.nn.Embedding(field_dims , output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim, )))

    def forward(self, x):
        # 전처리 다 끝나서 걍 넘겨주면 됨 . tensor type으로 넘어옴 .
        # 대신에 연속형변수 떼어서 넣줘야함 . + 연속형변수 크기 할당해두기.
        # 지금 하드코딩으로 박았음 . # emb에서 찾을때 long 아니면 에러남 . 9 ,3

        tensor_idx = x[:,:9]
        tensor_idx = tensor_idx.to(torch.long)

        tensor_continue = x[:,9:]
        tensor_continue = tensor_continue.unsqueeze(2)


        emb_value = self.fc(tensor_idx)

        emb_value[:,-3:,:] *=  tensor_continue


        #emb_value[:,-3:] = emb_value[:,-3:] * tensor_continue

        output = torch.sum(emb_value, dim = 1 ) + self.bias

        return output


class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, field_dims , embed_dims):
        super().__init__()
        self.feature_embedding = torch.nn.Embedding(field_dims ,embed_dims )
        torch.nn.init.xavier_uniform_(self.feature_embedding.weight.data)

    def forward(self, x):
        x = x[:,:9]
        x = x.to(torch.long)
        selected_embedding = self.feature_embedding(x)

        return selected_embedding

class FactorizationMachine(torch.nn.Module):
    def __init__(self , reduced_sum = True):
        super().__init__()
        self.reduced_sum = reduced_sum


    def forward(self , x):
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduced_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


