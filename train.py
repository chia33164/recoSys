from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
import utils
import collections
import sys

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

def bprloss(preds, vals, theta):
    pqu, ppi, nqu, npi = theta
    sig = torch.nn.Sigmoid()
    lamb = 0.002
    return (-1) * (torch.log(preds)).sum()
    # return (-1) * (torch.log(sig(preds)) + (lamb / 4) * (pqu.pow(2) + ppi.pow(2) + nqu.pow(2) + npi.pow(2))).sum()

class BPRModel(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors = 40, sparse=True, model=BaseModel):
        super(BPRModel, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.sparse = sparse
        self.pred_model = model(
            self.n_users,
            self.n_items,
            n_factors=n_factors,
            sparse=sparse
        )

    def forward(self, users, items):
        assert isinstance(items, tuple), \
            'Must pass in items as (pos_items, neg_items)'

        (pos_items, neg_items) = items

        acc = torch.empty(0)
        pos_preds, pqu, ppi = self.pred_model(users, pos_items)
        neg_items = neg_items.T
        for item in neg_items:
            neg_preds, nqu, npi = self.pred_model(users, item)
            acc = torch.cat((acc, pos_preds - neg_preds), 0)

        return (acc, (pqu, ppi, nqu, npi))

    def predict(self, users, items):
        return self.pred_model(users, items)

def bceloss(preds, vals, theta):
    pos, neg = preds
    sig = torch.nn.Sigmoid()
    posItem = sig(pos)
    negItem = sig(neg)
    return (-1) * (torch.log10(posItem).sum() + torch.log10(1.0 - negItem).sum())

class BCEModel(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors = 40, sparse=True, model=BaseModel):
        super(BCEModel, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.sparse = sparse
        self.pred_model = model(
            self.n_users,
            self.n_items,
            n_factors=n_factors,
            sparse=self.sparse
        )

    def forward(self, users, items):
        assert isinstance(items, tuple), \
            'Must pass in items as (pos_items, neg_items)'

        (pos_items, neg_items) = items

        acc_p = torch.empty(0)
        acc_n = torch.empty(0)

        pos_preds, pqu, ppi = self.pred_model(users, pos_items)
        neg_items = neg_items.T
        for item in neg_items:
            neg_preds, nqu, npi = self.pred_model(users, item)
            acc_p = torch.cat((acc_p, pos_preds), 0)
            acc_n = torch.cat((acc_n, neg_preds), 0)

        return (acc_p, acc_n), (pqu, ppi, nqu, npi)

    def predict(self, users, items):
        return self.pred_model(users, items)

class Interactions(torch.utils.data.Dataset):
    def __init__(self, mat):
        self.mat = mat.astype(np.float32).tocoo()
        self.n_users = self.mat.shape[0]
        self.n_items = self.mat.shape[1]

    def __getitem__(self, index):
        row = self.mat.row[index]
        col = self.mat.col[index]
        val = self.mat.data[index]
        return (row, col), val

    def __len__(self):
        return self.mat.nnz

class PairwiseInteractions(torch.utils.data.Dataset):

    def __init__(self, mat, neg_rate=1.0):
        self.mat = mat.astype(np.float32).tocoo()

        self.n_users = self.mat.shape[0]
        self.n_items = self.mat.shape[1]
        self.neg_rate = neg_rate

        self.mat_csr = self.mat.tocsr()
        if not self.mat_csr.has_sorted_indices:
            self.mat_csr.sort_indices()

    def __getitem__(self, index):
        row = self.mat.row[index]
        neg_cols = self.get_neg_col(row, rate=self.neg_rate)
        pos_cols = self.mat.col[index]
        val = self.mat.data[index]
        return (row, (pos_cols, neg_cols)), val

    def __len__(self):
        return self.mat.nnz

    def get_neg_col(self, row, rate=1.0):
        pos_cols = self.get_row_indices(row)
        full_cols = np.arange(self.n_items)
        missing_data = np.delete(full_cols, pos_cols)
        neg_cols = np.random.choice(missing_data, int(rate))
        return neg_cols

    def get_row_indices(self, row):
        start = self.mat_csr.indptr[row]
        end = self.mat_csr.indptr[row + 1]
        return self.mat_csr.indices[start:end]

class Run:
    def __init__(self,
                 train_matrix,
                 test_matrix,
                 test_data,
                 model=BaseModel,
                 n_factors = 40,
                 batch_size = 32,
                 sparse = True,
                 lr=0.1,
                 optimizer=torch.optim.SGD,
                 loss_function=torch.nn.MSELoss(reduction='sum'),
                 epoches = 10,
                 interaction_class=Interactions,
                 weight_decay=0.01,
                 num_workers = 2,
                 neg_rate = 1.0,
                 momentum=0,
                 eval_metrics=None):
        self.train_matrix = train_matrix
        self.test_matrix = test_matrix
        self.n_users = train_matrix.shape[0]
        self.n_items = train_matrix.shape[1]
        self.n_factors = n_factors
        self.batch_size = batch_size
        self.sparse = sparse
        self.lr = lr
        self.num_workers = num_workers
        self.epoches = epoches
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.neg_rate = neg_rate
        self.test_data = test_data
        self.model = model(self.n_users, self.n_items, n_factors=self.n_factors, sparse=self.sparse)
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        self.loss_function = loss_function
        self.train_loader = torch.utils.data.DataLoader(interaction_class(train_matrix, self.neg_rate), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        if test_matrix is not None:
            self.test_loader = torch.utils.data.DataLoader(interaction_class(test_matrix, 1.0), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.losses = collections.defaultdict(list)
        if eval_metrics is None:
            eval_metrics = []
        self.eval_metrics = eval_metrics

    def fit(self):
        for epoch in range(1, self.epoches + 1):
            self.model.train()
            total_loss = torch.Tensor([0])
            pbar = tqdm(enumerate(self.train_loader),
                        total=len(self.train_loader),
                        desc='({0:^3})'.format(epoch))
            for idx, ((row, col), val) in pbar:
                self.optimizer.zero_grad()

                row = row.long()
                if isinstance(col, list):
                    col = tuple(c.long() for c in col)
                else:
                    col = col.long()
                val = val.float()

                pred, theta = self.model(row, col)
                loss = self.loss_function(pred, val, theta)
                loss.backward()

                self.optimizer.step()

                total_loss += loss.item()
                batch_loss = loss.item() / (row.size()[0] * (self.neg_rate + 1))
                pbar.set_postfix(train_loss=batch_loss)
            total_loss /= (self.train_matrix.nnz * (self.neg_rate + 1))
            train_loss = total_loss[0]
            self.losses['train'].append(train_loss)
            row = 'Epoch: {0:^3}  train: {1:^10.5f}'.format(epoch, self.losses['train'][-1])

            ## testing
            if self.test_matrix is not None:
                self.losses['test'].append(self._validation_loss())
                row += 'val: {0:^10.5f}'.format(self.losses['test'][-1])
                row += 'map: {0:^10.5f}'.format(self._map())
            self.losses['epoch'].append(epoch)
            print(row)
        self.generate_output()

    def _validation_loss(self):
        self.model.eval()
        total_loss = torch.Tensor([0])
        for idx, ((row, col), val) in enumerate(self.test_loader):
            row = row.long()
            if isinstance(col, list):
                col = tuple(c.long() for c in col)
            else:
                col = col.long()
            val = val.float()

            pred, theta = self.model(row, col)
            loss = self.loss_function(pred, val, theta)
            total_loss += loss.item()

        total_loss /= self.test_matrix.nnz
        return total_loss[0]

    def _map(self):
        self.model.eval()
        df = {
            'UserId': [],
            'ItemId': []
        }
        top_k = 50
        for user in range(self.n_users):
            zeros = (self.train_matrix.row == user)
            zeros = self.train_matrix.col[zeros]
            UserIds = torch.Tensor([user] * self.n_items).int()
            ItemIds = torch.Tensor(np.arange(0, self.n_items, 1)).int()
            preds, qu, pi = self.model.predict(UserIds, ItemIds)
            preds = preds.cpu().detach().numpy()
            preds[zeros] = -100
            top_k_idx = preds.argsort()[::-1][0:top_k].astype(np.str)
            df['UserId'].extend([user])
            df['ItemId'].extend([" ".join(top_k_idx)])

        total = 0
        # MAP, Mean Average Precision
        for i in range(1, len(self.test_data)):
            precision_accu = 0
            correct = 0
            we_tmp = df['ItemId'][i - 1].split()
            ans_tmp = self.test_data[i][1].split()
            for j in range(0, len(we_tmp)):
                if we_tmp[j] in ans_tmp:
                    correct += 1
                    precision_accu += correct / (j + 1)
            total += precision_accu / len(ans_tmp)
        MAP = total / (self.n_users)
        return MAP

    def generate_output(self):
        self.save_model()
        self.model.eval()
        df = {
            'UserId': [],
            'ItemId': []
        }
        top_k = 50
        for user in range(self.n_users):
            zeros = (self.train_matrix.row == user)
            zeros = self.train_matrix.col[zeros]
            UserIds = torch.Tensor([user] * self.n_items).int()
            ItemIds = torch.Tensor(np.arange(0, self.n_items, 1)).int()
            preds, qu, pi = self.model.predict(UserIds, ItemIds)
            preds = preds.cpu().detach().numpy()
            preds[zeros] = -100
            top_k_idx = preds.argsort()[::-1][0:top_k].astype(np.str)
            df['UserId'].extend([user])
            df['ItemId'].extend([" ".join(top_k_idx)])
        df = pd.DataFrame(df, columns = ['UserId', 'ItemId'])
        df.to_csv(sys.argv[1] + '.csv', index=0)

    def save_model(self):
        print('Save model...')
        torch.save(self.model.pred_model.state_dict(), sys.argv[1] + '.pt')
        print('Complete')

def main():
    if len(sys.argv) != 2:
        print("Please enter output.csv")
    train_matrix, test_matrix = utils.get_dataset_from_local()
    test_data = utils.get_test()
    train_matrix = utils.get_all()
    # run = Run(train_matrix, test_matrix)
    # run = Run(train_matrix, test_matrix, test_data, model=BPRModel, sparse=False, momentum=0, batch_size=1024, n_factors=64, loss_function=bprloss, epoches=10, eval_metrics=None, interaction_class=PairwiseInteractions, lr=0.05, weight_decay=0, neg_rate=15.0)
    run = Run(train_matrix, None, test_data, model=BCEModel, sparse=False, momentum=0.1, batch_size=4096, n_factors=128, loss_function=bceloss, epoches=1, eval_metrics=None, interaction_class=PairwiseInteractions, lr=0.01, weight_decay=0, neg_rate=15.0)
    run.fit()

if __name__ == '__main__':
    main()
