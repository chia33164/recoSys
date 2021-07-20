import sys
import torch
import utils
import numpy as np
import pandas as pd

class BaseModel(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors = 40, sparse=True):
        super(BaseModel, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.sparse = sparse

        self.user_biases = torch.nn.Embedding(n_users, 1, sparse=sparse)
        self.item_biases = torch.nn.Embedding(n_items, 1, sparse=sparse)
        self.user_factors = torch.nn.Embedding(n_users, n_factors, sparse=sparse)
        self.item_factors = torch.nn.Embedding(n_items, n_factors, sparse=sparse)
        torch.nn.init.normal_(self.user_factors.weight, std=0.01)
        torch.nn.init.normal_(self.item_factors.weight, std=0.01)

    def forward(self, user, item):
        user_matrix = self.user_factors(user)
        item_matrix = self.item_factors(item)
        # pred  = self.user_biases(user).sum(1)
        # pred += self.item_biases(item).sum(1)
        pred = (user_matrix * item_matrix).sum(1)
        qu = user_matrix.pow(2).sum(1)
        pi = item_matrix.pow(2).sum(1)
        return pred, qu, pi

    def predict(self, user, item):
        return self.forward(user, item)

train_matrix = utils.get_all()
n_users, n_items = train_matrix.shape[0], train_matrix.shape[1]

PATH = 'best.pt'
model = BaseModel(n_users, n_items, n_factors=128, sparse=False)
model.load_state_dict(torch.load(PATH))
model.eval()
print(model)


df = {
    'UserId': [],
    'ItemId': []
}
top_k = 50
for user in range(n_users):
    zeros = (train_matrix.row == user)
    zeros = train_matrix.col[zeros]
    UserIds = torch.Tensor([user] * n_items).int()
    ItemIds = torch.Tensor(np.arange(0, n_items, 1)).int()
    preds, qu, pi = model.predict(UserIds, ItemIds)
    preds = preds.cpu().detach().numpy()
    preds[zeros] = -100
    top_k_idx = preds.argsort()[::-1][0:top_k].astype(np.str)
    df['UserId'].extend([user])
    df['ItemId'].extend([" ".join(top_k_idx)])
df = pd.DataFrame(df, columns = ['UserId', 'ItemId'])
df.to_csv(sys.argv[1] + '.csv', index=0)

