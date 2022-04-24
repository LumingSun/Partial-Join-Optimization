import torch
import torch.optim
import joblib
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from torch.utils.data import DataLoader
import net
from featurize import TreeFeaturizer
import json
import sys
import time
import os
import storage
import model
import train
import math
import reg_blocker
import numpy as np

PG_OPTIMIZER_INDEX = 0
DEFAULT_MODEL_PATH = "bao_default_model"
TMP_MODEL_PATH = "bao_tmp_model"
OLD_MODEL_PATH = "bao_previous_model"

CUDA = torch.cuda.is_available()

def _nn_path(base):
    return os.path.join(base, "nn_weights")

def _x_transform_path(base):
    return os.path.join(base, "x_transform")

def _y_transform_path(base):
    return os.path.join(base, "y_transform")

def _channels_path(base):
    return os.path.join(base, "channels")

def _n_path(base):
    return os.path.join(base, "n")


def _inv_log1p(x):
    return np.exp(x) - 1

class BaoData:
    def __init__(self, data):
        assert data
        self.__data = data

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx):
        return (self.__data[idx]["tree"],
                self.__data[idx]["target"])

def collate(x):
    trees = []
    targets = []

    for tree, target in x:
        trees.append(tree)
        targets.append(target)

    targets = torch.tensor(np.array(targets))
    return trees, targets

class BaoRegression:
    def __init__(self, verbose=False, have_cache_data=False):
        self.__net = None
        self.__verbose = verbose

        log_transformer = preprocessing.FunctionTransformer(
            np.log1p, _inv_log1p,
            validate=True)
        scale_transformer = preprocessing.MinMaxScaler()

        self.__pipeline = Pipeline([("log", log_transformer),
                                    ("scale", scale_transformer)])
        
        self.__tree_transform = TreeFeaturizer()
        self.__have_cache_data = have_cache_data
        self.__in_channels = None
        self.__n = 0
        
    def __log(self, *args):
        if self.__verbose:
            print(*args)

    def num_items_trained_on(self):
        return self.__n
            
    def load(self, path):
        with open(_n_path(path), "rb") as f:
            self.__n = joblib.load(f)
        with open(_channels_path(path), "rb") as f:
            self.__in_channels = joblib.load(f)
            
        self.__net = net.BaoNet(self.__in_channels)
        self.__net.load_state_dict(torch.load(_nn_path(path)))
        self.__net.eval()
        
        with open(_y_transform_path(path), "rb") as f:
            self.__pipeline = joblib.load(f)
        with open(_x_transform_path(path), "rb") as f:
            self.__tree_transform = joblib.load(f)

    def save(self, path):
        # try to create a directory here
        os.makedirs(path, exist_ok=True)
        
        torch.save(self.__net.state_dict(), _nn_path(path))
        with open(_y_transform_path(path), "wb") as f:
            joblib.dump(self.__pipeline, f)
        with open(_x_transform_path(path), "wb") as f:
            joblib.dump(self.__tree_transform, f)
        with open(_channels_path(path), "wb") as f:
            joblib.dump(self.__in_channels, f)
        with open(_n_path(path), "wb") as f:
            joblib.dump(self.__n, f)

    def fit(self, X, y):
        if isinstance(y, list):
            y = np.array(y)

        X = [json.loads(x) if isinstance(x, str) else x for x in X]
        self.__n = len(X)
            
        # transform the set of trees into feature vectors using a log
        # (assuming the tail behavior exists, TODO investigate
        #  the quantile transformer from scikit)
        y = self.__pipeline.fit_transform(y.reshape(-1, 1)).astype(np.float32)
        
        self.__tree_transform.fit(X)
        X = self.__tree_transform.transform(X)

        pairs = list(zip(X, y))
        dataset = DataLoader(pairs,
                             batch_size=16,
                             shuffle=True,
                             collate_fn=collate)

        # determine the initial number of channels
        for inp, _tar in dataset:
            in_channels = inp[0][0].shape[0]
            break

        self.__log("Initial input channels:", in_channels)

        if self.__have_cache_data:
            assert in_channels == self.__tree_transform.num_operators() + 3
        else:
            assert in_channels == self.__tree_transform.num_operators() + 2

        self.__net = net.BaoNet(in_channels)
        self.__in_channels = in_channels
        if CUDA:
            self.__net = self.__net.cuda()

        optimizer = torch.optim.Adam(self.__net.parameters())
        loss_fn = torch.nn.MSELoss()
        
        losses = []
        for epoch in range(100):
            loss_accum = 0
            for x, y in dataset:
                if CUDA:
                    y = y.cuda()
                y_pred = self.__net(x)
                # FIXME: loss change
                loss = loss_fn(y_pred, y)
                # loss = self.__net.sample_elbo(inputs=x, labels=y, criterion=loss_fn, sample_nbr=3)
                loss_accum += loss.item()
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_accum /= len(dataset)
            losses.append(loss_accum)
            if epoch % 15 == 0:
                self.__log("Epoch", epoch, "training loss:", loss_accum)

            # stopping condition
            if len(losses) > 10 and losses[-1] < 0.1:
                last_two = np.min(losses[-2:])
                if last_two > losses[-10] or (losses[-10] - last_two < 0.0001):
                    self.__log("Stopped training from convergence condition at epoch", epoch)
                    break
        else:
            self.__log("Stopped training after max epochs")

    def predict(self, X, sample_nbr=100):
        if not isinstance(X, list):
            X = [X]
        X = [json.loads(x) if isinstance(x, str) else x for x in X]

        X = self.__tree_transform.transform(X)
        
        self.__net.eval()
        preds = np.array([self.__net(X).cpu().detach().numpy().flatten() for i in range(sample_nbr)])
        pred = np.mean(preds,axis=0).reshape(-1,1)
        return self.__pipeline.inverse_transform(pred), self.__pipeline.inverse_transform(preds)

def add_buffer_info_to_plans(buffer_info, plans):
    for p in plans:
        p["Buffers"] = buffer_info
    return plans


class Model:
    def __init__(self):
        self.__current_model = None
        
    def confidence(self, preds, sample_nbr=100, std_multiplier=2):
        mean = np.mean(preds, axis=0)
        std = np.std(preds, axis=0)
        ci_upper = mean + (std_multiplier * std)
        ci_lower = mean - (std_multiplier * std)
        # print("mean: ", mean)
        # print("std: ",std)
        # print("preds: ", preds)
        # print("upper: ", ci_upper)
        # print("lower: ", ci_lower)
        in_confidence = []
        for dim in range(5):
            cnt = 0
            for pred in preds:
                if(pred[dim]<=ci_upper[dim] and pred[dim]>=ci_lower[dim]):
                    cnt += 1
            in_confidence.append(cnt/sample_nbr)
        return in_confidence
    
    def select_plan(self, messages):
        start = time.time()
        # the last message is the buffer state
        *arms, buffers = messages
        # if we don't have a model, default to the PG optimizer
        if self.__current_model is None:
            return PG_OPTIMIZER_INDEX

        # if we do have a model, make predictions for each plan.
        arms = add_buffer_info_to_plans(buffers, arms)
        res, preds = self.__current_model.predict(arms)
        # save estimated cost of different arms, this is for hint selection page
        # only need to save for current sql
        # with open("/home/slm/pg_related/BaoForPostgreSQL/query_log/arm_cost.txt","w") as f:
            # f.writelines("\n".join(["%.2f" % x for x in res.flatten()]))

        idx = res.argmin()
        
        confidence = self.confidence(preds, std_multiplier=1.5)
        # with open("/home/slm/pg_related/BaoForPostgreSQL/query_log/confidence.txt","w") as f:
        #     f.writelines("\n".join(["%.2f" % x for x in confidence]))
            
        # save selected plan
        # sql_count = len(os.listdir("/home/slm/pg_related/BaoForPostgreSQL/query_log/plan_log/"))
        # with open("/home/slm/pg_related/BaoForPostgreSQL/query_log/plan_log/{}.csv".format(sql_count),"w") as f:
        #     json.dump(arms[idx], f, ensure_ascii=False)
            
        stop = time.time()
        print("Selected index", idx,
              "after", f"{round((stop - start) * 1000)}ms",
              "Predicted reward / PG:", res[idx][0],
              "/", res[0][0])
        return idx

    def predict(self, messages):
        # the last message is the buffer state
        plan, buffers = messages

        # if we don't have a model, make a prediction of NaN
        if self.__current_model is None:
            return math.nan

        # if we do have a model, make predictions for each plan.
        plans = add_buffer_info_to_plans(buffers, [plan])
        res, _ = self.__current_model.predict(plans)
        return res[0][0]
    
    def load_model(self, fp):
        try:
            new_model = model.BaoRegression(have_cache_data=True)
            new_model.load(fp)

            if reg_blocker.should_replace_model(
                    self.__current_model,
                    new_model):
                self.__current_model = new_model
                print("Accepted new model.")
            else:
                print("Rejecting load of new model due to regresison profile.")
                
        except Exception as e:
            print("Failed to load Bao model from", fp,
                  "Exception:", sys.exc_info()[0])
            raise e