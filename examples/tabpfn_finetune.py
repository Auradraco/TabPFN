## Debug TabPFN on Adult to better understand training and inference.
import sklearn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
from tabpfn.utils import collate_for_tabpfn_dataset
import json
from functools import partial
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam


def eval_test(clf, my_dl_test, lossfn):
    with torch.no_grad():
        loss_sum = 0.0
        acc_sum = 0.0
        acc_items = 0
        for i, data_batch in enumerate(my_dl_test):
            X_trains, X_tests, y_trains, y_tests, cat_ixs, confs = data_batch
            clf.fit_from_preprocessed(X_trains, y_trains, cat_ixs, confs)
            preds = clf.predict_proba_from_preprocessed(X_tests)
            loss_sum += lossfn(torch.log(preds), y_tests.to(device)).item()
            acc_sum += accuracy_score(y_tests.flatten().cpu(), preds[:,1,:].flatten().cpu()>0.5)*y_tests.numel()
            acc_items += y_tests.numel()
        
        res_accuracy = acc_sum/acc_items
        print("Test Acc:", res_accuracy)
        print("Test Loss:", loss_sum)
        
        return loss_sum, res_accuracy
    
if __name__ == "__main__":
    device = "cuda:1"
    n_use = 200_000
    do_epochs = 1

    # Load Covertype Dataset (7-way classification)
    data_frame_x, data_frame_y = sklearn.datasets.fetch_covtype(return_X_y=True, shuffle=True)
    

    splitfn = partial(train_test_split, test_size=0.3)
    X_train, X_test, y_train, y_test = splitfn(data_frame_x[:n_use], data_frame_y[:n_use])

    clf = TabPFNClassifier(ignore_pretraining_limits=True, device=device, n_estimators=2, 
                           random_state=2, inference_precision=torch.float32)
    
    datasets_list = clf.get_preprocessed_datasets(X_train, y_train, splitfn, 1000)
    datasets_list_test = clf.get_preprocessed_datasets(X_test, y_test, splitfn, 1000)
    my_dl_train = DataLoader(datasets_list, batch_size=2, collate_fn=collate_for_tabpfn_dataset)
    my_dl_test = DataLoader(datasets_list_test, batch_size=1, collate_fn=collate_for_tabpfn_dataset)
    
    optim_impl = Adam(clf.model_.parameters(), lr=1e-5)
    lossfn = torch.nn.NLLLoss()
    loss_batches = []
    acc_batches = []
    steps = 0

    loss_test, res_acc = eval_test(clf, my_dl_test, lossfn)
    print("Initial accuracy:", res_acc)
    for epochs in range(1):
        for data_batch in tqdm(my_dl_train):
            optim_impl.zero_grad()
            X_trains, X_tests, y_trains, y_tests, cat_ixs, confs = data_batch
            clf.fit_from_preprocessed(X_trains, y_trains, cat_ixs, confs)
            preds = clf.predict_proba_from_preprocessed(X_tests)
            loss = lossfn(torch.log(preds), y_tests.to(device))
            loss.backward()
            optim_impl.step()
        steps += 1
        
        loss_test, res_acc = eval_test(clf, my_dl_test, lossfn)
        loss_batches.append(loss_test)
        acc_batches.append(res_acc)
        json.dump({"loss": loss_batches, "acc": acc_batches}, open("finetune.json", "w"))

            
        