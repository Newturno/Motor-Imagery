import copy
import numpy as np
import torch
from torch.autograd import Variable
import torch.cuda as cuda
import os
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader



def setup_dataflow(X_tensor,y_tensor, train_idx, val_idx):
    #train_sampler = SubsetRandomSampler(train_idx)
    #val_sampler = SubsetRandomSampler(val_idx)
    train_datasets = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
    val_datasets = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])
    
    train_loader = DataLoader(train_datasets, batch_size=32)
    val_loader = DataLoader(val_datasets, batch_size=32)

    return train_loader, val_loader

def create_dataloader(X, y, batch_size):
    X_tensor = torch.tensor(X).float()
    y_tensor = torch.tensor(y).long()
    dataset_tensor = TensorDataset(X_tensor, y_tensor)
    dl = torch.utils.data.DataLoader(dataset_tensor, batch_size=batch_size, shuffle=True)
    return dl

def train(model, loader_train, loader_test, optimizer,criterion,device,wand,vail_loader= None,
          cross = False):
    # put model on cuda if not already
    device = torch.device(device)
    config = wand.config
    
    weights_name = config.weightname
    best_val_loss = + np.infty
    best_model = copy.deepcopy(model)
    train_loss = []
    valid_loss = [10,11]
    train_accuracy = []
    valid_accuracy = []
    
    old_loss = 100
    old_acc = 0
    valid_loss_vail = []
    
    for epoch in range(config.epochs):
        iter_loss = 0.0
        correct = 0
        iterations = 0

        model.train()

        for i, (items, classes) in enumerate(loader_train):
            items = Variable(items)
            classes = classes.type(torch.LongTensor)
            classes = Variable(classes)

            if cuda.is_available():
                items = items.to(device=device)
                classes = classes.to(device=device)

            optimizer.zero_grad()
            outputs = model(items)
            loss = criterion(outputs, classes)

            iter_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            metrics = {"train/train_loss": loss}
            if i + 1 < config.num_step_per_epoch:
                # 🐝 Log train metrics to wandb 
                wand.log(metrics)
            
            #print(loss)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == classes.data).sum()
            iterations += 1

        train_loss.append(iter_loss/iterations)
        

        train_accuracy.append((100 * correct.float() / len(loader_train.dataset)))
        train_metrics = {"train/train_loss": iter_loss/iterations, 
                       "train/train_accuracy": (100 * correct.float() / len(loader_train.dataset))}
        
        wand.log({**metrics, **train_metrics})

        loss = 0.0
        correct = 0
        iterations = 0
        model.eval()
        
        for i, (items, classes) in enumerate(loader_test):
            items = Variable(items)
            classes = Variable(classes)
            
            if cuda.is_available():
                items = items.to(device=device)
                classes = classes.to(device=device)
            
            outputs = model(items)
            loss += criterion(outputs, classes).item()
            
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == classes.data).sum()
            
            iterations += 1
        
        valid_loss.append(loss/iterations)
        correct_scalar = np.array([correct.clone().cpu()])[0]
        valid_accuracy.append(correct_scalar / len(loader_test.dataset) * 100.0)
        
        val_metrics = {"val/val_loss": loss/iterations, 
                       "val/val_accuracy": correct_scalar / len(loader_test.dataset) * 100.0}
        wand.log({**metrics, **val_metrics})

        epoch_acc = correct.double()/len(loader_test.dataset)
        #and old_acc <= valid_accuracy[-1]
        if epoch+1 > 2 and valid_loss[-1] < old_loss :
                newpath = r'./save_weight/{}'.format(weights_name) 
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                torch.save(model.state_dict(),'./save_weight/{}/{:.4f}_{}_{:.4f}_{:.4f}.pth'.format(weights_name,valid_loss[-1],weights_name,valid_loss[-1],valid_accuracy[-1]))
                old_loss = valid_loss[-1]  
                old_acc = valid_accuracy[-1]
        if (epoch % 100) ==0:
            print ('Epoch %d/%d, Tr Loss: %.4f, Tr Acc: %.4f, Val Loss: %.4f, Val Acc: %.4f'
                       %(epoch+1, config.epochs, train_loss[-1], train_accuracy[-1], valid_loss[-1], valid_accuracy[-1]))
         
        if cross :
                loss_vail = 0.0
                correct_vail = 0
                iterations_vail = 0
                model.eval()

                for i, (items, classes) in enumerate(vail_loader):
                    classes = classes.type(torch.LongTensor)
                    items = Variable(items)
                    classes = Variable(classes)

                    if cuda.is_available():
                        items = items.to(device=device)
                        classes = classes.to(device=device)


                    outputs = model(items)
                    loss_vail += criterion(outputs, classes).item()

                    _, predicted = torch.max(outputs.data, 1)

                    correct_vail += (predicted == classes.data).sum()
                    #print("correct : {}".format(classes.data))
                    #print("predicted : {}".format(predicted))
                    iterations_vail += 1

                valid_loss_vail.append(loss_vail/iterations_vail)
                correct_scalar = np.array([correct_vail.clone().cpu()])[0]
                valid_accuracy.append(correct_scalar / len(vail_loader.dataset) * 100.0)
                vali_metrics = {"val/val_loss": loss_vail/iterations, 
                        "val/val_accuracy": correct_scalar / len(loader_test.dataset) * 100.0}
                wand.log({**metrics, **vali_metrics})
                if (epoch % 100) ==0:
                    print ('Val Loss: {0}, Val Acc: {1}'.format(valid_loss_vail[-1], valid_accuracy[-1]))

    return train_loss,valid_loss,train_accuracy,valid_accuracy