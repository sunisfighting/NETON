import numpy as np
import functools

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from abc import ABC, abstractmethod
from utils import get_split, split_dataset


def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, function_name):
    print("======== Final ========")
    print(f'(E) ||{function_name}||:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}'+u"\u00B1"+f'{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()


@repeat(20)
def node_classification(embeddings, y, ratio=0.1):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(bool)
    X = normalize(X, norm='l2')
    X_tv, X_test, Y_tv, Y_test = train_test_split(X, Y, test_size=1 - 2*ratio) # 0.8 for test
    X_train, X_val, Y_train, Y_val = train_test_split(X_tv, Y_tv, test_size=0.5) # 0.1 for train and 0.1 for val

    c = 2.0 ** np.arange(-10, 10)
    best_score = 0
    best_logreg = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=c[0]))
    for c_i in c:
        logreg = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=c_i))
        logreg.fit(X_train, Y_train)
        score = logreg.score(X_val, Y_val)
        if score > best_score:
            best_score = score
            best_logreg = logreg

    Y_pred = best_logreg.predict_proba(X_test)
    Y_pred = prob_to_one_hot(Y_pred)

    micro = f1_score(Y_test, Y_pred, average="micro")
    macro = f1_score(Y_test, Y_pred, average="macro")
    print(f'(E): Best test F1Mi={micro:.4f}, F1Ma={macro:.4f}')

    return {
        'F1Mi': micro,
        'F1Ma': macro
    }


def node_classification_citation(embeddings, y, ratio=0.1, train_mask=None, val_mask=None, test_mask=None):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(bool)
    X = normalize(X, norm='l2')
    if train_mask != None and val_mask != None and test_mask!= None:
        train_mask = train_mask.detach().cpu().numpy()
        val_mask = val_mask.detach().cpu().numpy()
        test_mask = test_mask.detach().cpu().numpy()
        X_train, Y_train = X[train_mask], Y[train_mask]
        X_val, Y_val = X[val_mask], Y[val_mask]
        X_test, Y_test = X[test_mask], Y[test_mask]
    else:
        X_tv, X_test, Y_tv, Y_test = train_test_split(X, Y, test_size=1 - 2*ratio) # 0.8 for test
        X_train, X_val, Y_train, Y_val = train_test_split(X_tv, Y_tv, test_size=0.5) # 0.1 for train and 0.1 for val

    c = 2.0 ** np.arange(-10, 10)
    best_score = 0
    best_logreg = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=c[0]))
    for c_i in c:
        logreg = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=c_i))
        logreg.fit(X_train, Y_train)
        score = logreg.score(X_val, Y_val)
        if score > best_score:
            best_score = score
            best_logreg = logreg

    Y_pred = best_logreg.predict_proba(X_test)
    Y_pred = prob_to_one_hot(Y_pred)
    # Y_pred = np.argmax(Y_pred, axis =1)
    micro = f1_score(Y_test, Y_pred, average="micro")
    macro = f1_score(Y_test, Y_pred, average="macro")
    print(f'(E): Best test F1Mi={micro:.4f}, F1Ma={macro:.4f}')

    return {
        'F1Mi': micro,
        'F1Ma': macro
    }

def node_classification_wiki(embeddings, data=None):
    X = embeddings.detach().cpu().numpy()
    Y = data.y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)
    X = normalize(X, norm='l2')
    assert data != None
    results = []
    for inx in range(data.train_mask.shape[1]):
        split = split_dataset(data, 'wikics', split_idx=inx)
        trn_split = split['train'].reshape(-1).detach().cpu().numpy()
        val_split = split['val'].reshape(-1).detach().cpu().numpy()
        tst_split = split['test'].reshape(-1).detach().cpu().numpy()
        X_train, Y_train = X[trn_split], Y[trn_split]
        X_val, Y_val = X[val_split], Y[val_split]
        X_test, Y_test = X[tst_split], Y[tst_split]

        c = 2.0 ** np.arange(-10, 10)
        best_score = 0
        best_logreg = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=c[0]))
        for c_i in c:
            logreg = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=c_i))
            logreg.fit(X_train, Y_train)
            score = logreg.score(X_val, Y_val)
            if score > best_score:
                best_score = score
                best_logreg = logreg
        Y_pred = best_logreg.predict_proba(X_test)
        Y_pred = prob_to_one_hot(Y_pred)
        micro = f1_score(Y_test, Y_pred, average="micro")
        macro = f1_score(Y_test, Y_pred, average="macro")
        results.append({'F1Mi': micro, 'F1Ma': macro})
        print(f'(E): Best test F1Mi={micro:.4f}, F1Ma={macro:.4f}')
    statistics = {}
    for key in results[0].keys():
        values = [r[key] for r in results]
        statistics[key] = {
            'mean': np.mean(values),
            'std': np.std(values)}
    print_statistics(statistics, 'node_classification')

@repeat(3)
def node_classification_reddit(embeddings, data=None):
    X = embeddings.detach().cpu().numpy()
    Y = data.y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)
    X = normalize(X, norm='l2')
    assert data != None
    X_train, Y_train = X[data.train_mask], Y[data.train_mask]
    X_val, Y_val = X[data.val_mask], Y[data.val_mask]
    X_test, Y_test = X[data.test_mask], Y[data.test_mask]

    c = 2.0 ** np.arange(-10, 10)
    best_score = 0
    best_logreg = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=c[0]))
    for c_i in c:
        logreg = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=c_i))
        logreg.fit(X_train, Y_train)
        score = logreg.score(X_val, Y_val)
        if score > best_score:
            best_score = score
            best_logreg = logreg

    Y_pred = best_logreg.predict_proba(X_test)
    Y_pred = prob_to_one_hot(Y_pred)
    # Y_pred = np.argmax(Y_pred, axis =1)
    micro = f1_score(Y_test, Y_pred, average="micro")
    macro = f1_score(Y_test, Y_pred, average="macro")
    print(f'(E): Best test F1Mi={micro:.4f}, F1Ma={macro:.4f}')

    return {
        'F1Mi': micro,
        'F1Ma': macro
    }




#####################################################################
class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, z: torch.FloatTensor, data) -> dict:
        pass

    def evaluate_wikics(self, z: torch.FloatTensor, data) -> dict:
        pass

    def evaluate_citation(self, z: torch.FloatTensor, data) -> dict:
        pass

    def evaluate_reddit(self, z: torch.FloatTensor, data) -> dict:
        pass

    def evaluate_multiclass(self, z: torch.FloatTensor, y, split_idx) -> dict:
        pass

    def evaluate_arxiv(self, z: torch.FloatTensor, data, split_idx) -> dict:
        pass

    def __call__(self, z: torch.FloatTensor, data, wikics=False, citation = False, reddit=False, arxiv=False, multi_class=False, split_idx=None) -> dict:
        if wikics:
            result = self.evaluate_wikics(z, data)
        elif reddit:
            result = self.evaluate_reddit(z, data)
        elif multi_class:
            result = self.evaluate_multiclass(z, data, split_idx)
        elif arxiv:
            result = self.evaluate_arxiv(z, data, split_idx)
        elif citation:
            result = self.evaluate_citation(z, data)
        else:
            result = self.evaluate(z, data)
        return result

class LogisticReg(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticReg, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z

class LREvaluator(BaseEvaluator):
    def __init__(self, num_epochs: int = 5000, learning_rate: float = 0.1,
                 weight_decay: float = 0, test_interval: int = 20, train_ratio = 0.1, test_ratio = 0.8):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

    @repeat(20)
    def evaluate(self, z: torch.FloatTensor, data):
        split = get_split(num_samples=z.size()[0], train_ratio=self.train_ratio, test_ratio=self.test_ratio)
        device = z.device
        x = F.normalize(z.detach()).to(device)
        input_dim = x.size()[1]
        y = data.y.to(device)
        num_classes = y.max().item() + 1
        classifier = LogisticReg(input_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = nn.LogSoftmax(dim=-1)
        criterion = nn.NLLLoss()

        best_val_micro = 0
        best_test_micro = 0
        best_test_macro = 0
        best_epoch = 0

        with tqdm(total=self.num_epochs, desc='(LR)',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
            for epoch in range(self.num_epochs):
                classifier.train()
                optimizer.zero_grad()

                output = classifier(x[split['train']])
                loss = criterion(output_fn(output), y[split['train']])

                loss.backward()
                optimizer.step()

                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()
                    y_test = y[split['test']].detach().cpu().numpy()
                    y_pred = classifier(x[split['test']]).argmax(-1).detach().cpu().numpy()
                    test_micro = f1_score(y_test, y_pred, average='micro')
                    test_macro = f1_score(y_test, y_pred, average='macro')

                    y_val = y[split['val']].detach().cpu().numpy()
                    y_pred = classifier(x[split['val']]).argmax(-1).detach().cpu().numpy()
                    val_micro = f1_score(y_val, y_pred, average='micro')

                    if val_micro > best_val_micro:
                        best_val_micro = val_micro
                        best_test_micro = test_micro
                        best_test_macro = test_macro
                        best_epoch = epoch

                    pbar.set_postfix({'best test F1Mi': f'{best_test_micro:.4f}', 'F1Ma': f'{best_test_macro:.4f}'})
                    pbar.update(self.test_interval)

        return {
            'micro_f1': best_test_micro,
            'macro_f1': best_test_macro
        }

    def evaluate_wikics(self, z: torch.FloatTensor, data) -> dict:
        device = z.device
        x = F.normalize(z.detach()).to(device)
        input_dim = x.size()[1]
        y = data.y.to(device)
        num_classes = y.max().item() + 1

        results = []
        for inx in range(data.train_mask.shape[1]):
            classifier = LogisticReg(input_dim, num_classes).to(device)
            optimizer = Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            output_fn = nn.LogSoftmax(dim=-1)
            criterion = nn.NLLLoss()
            best_val_micro = 0
            best_test_micro = 0
            best_test_macro = 0
            best_epoch = 0
            split = split_dataset(data, 'wikics', split_idx=inx)

            with tqdm(total=self.num_epochs, desc='(LR)',bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
                for epoch in range(self.num_epochs):
                    classifier.train()
                    optimizer.zero_grad()

                    output = classifier(x[split['train']])
                    loss = criterion(output_fn(output), y[split['train']])

                    loss.backward()
                    optimizer.step()

                    if (epoch + 1) % self.test_interval == 0:
                        classifier.eval()
                        y_test = y[split['test']].detach().cpu().numpy()
                        y_pred = classifier(x[split['test']]).argmax(-1).detach().cpu().numpy()
                        test_micro = f1_score(y_test, y_pred, average='micro')
                        test_macro = f1_score(y_test, y_pred, average='macro')

                        y_val = y[split['val']].detach().cpu().numpy()
                        y_pred = classifier(x[split['val']]).argmax(-1).detach().cpu().numpy()
                        val_micro = f1_score(y_val, y_pred, average='micro')

                        if val_micro > best_val_micro:
                            best_val_micro = val_micro
                            best_test_micro = test_micro
                            best_test_macro = test_macro
                            best_epoch = epoch

                        pbar.set_postfix({'best test F1Mi': f'{best_test_micro:.4f}', 'F1Ma': f'{best_test_macro:.4f}'})
                        pbar.update(self.test_interval)
                results.append({'F1Mi': best_test_micro, 'F1Ma': best_test_macro})
        statistics = {}
        for key in results[0].keys():
            values = [r[key] for r in results]
            statistics[key] = {
                'mean': np.mean(values),
                'std': np.std(values)}
        print_statistics(statistics, 'node_classification')

    # citation
    def evaluate_citation(self, z: torch.FloatTensor, data):

        device = z.device
        x = F.normalize(z.detach()).to(device)
        input_dim = x.size()[1]
        y = data.y.to(device)
        num_classes = y.max().item() + 1
        classifier = LogisticReg(input_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = nn.LogSoftmax(dim=-1)
        criterion = nn.NLLLoss()

        best_val_micro = 0
        best_test_micro = 0
        best_test_macro = 0
        best_epoch = 0

        with tqdm(total=self.num_epochs, desc='(LR)',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
            for epoch in range(self.num_epochs):
                classifier.train()
                optimizer.zero_grad()

                output = classifier(x[data.train_mask])
                loss = criterion(output_fn(output), y[data.train_mask])

                loss.backward()
                optimizer.step()

                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()
                    y_test = y[data.test_mask].detach().cpu().numpy()
                    y_pred = classifier(x[data.test_mask]).argmax(-1).detach().cpu().numpy()
                    test_micro = f1_score(y_test, y_pred, average='micro')
                    test_macro = f1_score(y_test, y_pred, average='macro')

                    y_val = y[data.val_mask].detach().cpu().numpy()
                    y_pred = classifier(x[data.val_mask]).argmax(-1).detach().cpu().numpy()
                    val_micro = f1_score(y_val, y_pred, average='micro')

                    if val_micro > best_val_micro:
                        best_val_micro = val_micro
                        best_test_micro = test_micro
                        best_test_macro = test_macro
                        best_epoch = epoch

                    pbar.set_postfix({'best test F1Mi': f'{best_test_micro:.4f}', 'F1Ma': f'{best_test_macro:.4f}'})
                    pbar.update(self.test_interval)

        return {
            'micro_f1': best_test_micro,
            'macro_f1': best_test_macro
        }

    @repeat(3)
    def evaluate_reddit(self, z: torch.FloatTensor, data):
        device = z.device
        x = F.normalize(z.detach()).to(device)
        input_dim = x.size()[1]
        y = data.y.to(device)
        num_classes = y.max().item() + 1
        classifier = LogisticReg(input_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = nn.LogSoftmax(dim=-1)
        criterion = nn.NLLLoss()

        best_val_micro = 0
        best_test_micro = 0
        best_test_macro = 0
        best_epoch = 0

        with tqdm(total=self.num_epochs, desc='(LR)',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
            for epoch in range(self.num_epochs):
                classifier.train()
                optimizer.zero_grad()

                output = classifier(x[data.train_mask])
                loss = criterion(output_fn(output), y[data.train_mask])

                loss.backward()
                optimizer.step()

                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()
                    y_test = y[data.test_mask].detach().cpu().numpy()
                    y_pred = classifier(x[data.test_mask]).argmax(-1).detach().cpu().numpy()
                    test_micro = f1_score(y_test, y_pred, average='micro')
                    test_macro = f1_score(y_test, y_pred, average='macro')

                    y_val = y[data.val_mask].detach().cpu().numpy()
                    y_pred = classifier(x[data.val_mask]).argmax(-1).detach().cpu().numpy()
                    val_micro = f1_score(y_val, y_pred, average='micro')

                    if val_micro > best_val_micro:
                        best_val_micro = val_micro
                        best_test_micro = test_micro
                        best_test_macro = test_macro
                        best_epoch = epoch

                    pbar.set_postfix({'best test F1Mi': f'{best_test_micro:.4f}', 'F1Ma': f'{best_test_macro:.4f}'})
                    pbar.update(self.test_interval)

        return {
            'micro_f1': best_test_micro,
            'macro_f1': best_test_macro
        }

    @repeat(1)
    def evaluate_multiclass(self, z: torch.FloatTensor, y, split_idx):
        device = z.device
        x = F.normalize(z.detach()).to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        num_classes = y.shape[1]

        classifier = LogisticReg(input_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = torch.nn.BCEWithLogitsLoss()

        best_val_micro = 0
        best_test_micro = 0
        best_test_macro = 0
        best_epoch = 0

        with tqdm(total=self.num_epochs, desc='(LR)',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
            for epoch in range(self.num_epochs):
                classifier.train()
                optimizer.zero_grad()

                output = classifier(x[:split_idx[0]])
                loss = criterion(output, y[:split_idx[0]])

                loss.backward()
                optimizer.step()

                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()
                    y_test = y[split_idx[1]:split_idx[2]].detach().cpu().numpy()

                    out_test = classifier(x[split_idx[1]:split_idx[2]]).detach()
                    y_pred = (out_test > 0).float().cpu().numpy()

                    test_micro = f1_score(y_test, y_pred, average='micro')
                    test_macro = f1_score(y_test, y_pred, average='macro')

                    y_val = y[split_idx[0]:split_idx[1]].detach().cpu().numpy()
                    out_val = classifier(x[split_idx[0]:split_idx[1]]).detach()
                    y_pred = (out_val > 0).float().cpu().numpy()
                    val_micro = f1_score(y_val, y_pred, average='micro')

                    if val_micro > best_val_micro:
                        best_val_micro = val_micro
                        best_test_micro = test_micro
                        best_test_macro = test_macro
                        best_epoch = epoch

                    pbar.set_postfix({'best test F1Mi': f'{best_test_micro:.4f}', 'F1Ma': f'{best_test_macro:.4f}'})
                    pbar.update(self.test_interval)

        return {
            'micro_f1': best_test_micro,
            'macro_f1': best_test_macro
        }

    @repeat(20)
    def evaluate_arxiv(self, z: torch.FloatTensor, data, split_idx):

        device = z.device
        split = split_idx.to(device)
        x = F.normalize(z.detach()).to(device)
        input_dim = x.size()[1]
        y = data.y.to(device)
        num_classes = y.max().item() + 1
        classifier = LogisticReg(input_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = nn.LogSoftmax(dim=-1)
        criterion = nn.NLLLoss()

        best_val_micro = 0
        best_test_micro = 0
        best_test_macro = 0
        best_epoch = 0

        with tqdm(total=self.num_epochs, desc='(LR)',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
            for epoch in range(self.num_epochs):
                classifier.train()
                optimizer.zero_grad()

                output = classifier(x[split['train']])
                loss = criterion(output_fn(output), y[split['train']])

                loss.backward()
                optimizer.step()

                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()
                    y_test = y[split['test']].detach().cpu().numpy()
                    y_pred = classifier(x[split['test']]).argmax(-1).detach().cpu().numpy()
                    test_micro = f1_score(y_test, y_pred, average='micro')
                    test_macro = f1_score(y_test, y_pred, average='macro')

                    y_val = y[split['valid']].detach().cpu().numpy()
                    y_pred = classifier(x[split['valid']]).argmax(-1).detach().cpu().numpy()
                    val_micro = f1_score(y_val, y_pred, average='micro')

                    if val_micro > best_val_micro:
                        best_val_micro = val_micro
                        best_test_micro = test_micro
                        best_test_macro = test_macro
                        best_epoch = epoch

                    pbar.set_postfix({'best test F1Mi': f'{best_test_micro:.4f}', 'F1Ma': f'{best_test_macro:.4f}'})
                    pbar.update(self.test_interval)

        return {
            'micro_f1': best_test_micro,
            'macro_f1': best_test_macro
        }

