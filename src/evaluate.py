import torch
from torch.utils.data import DataLoader

# to calculate train/test accuray
def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc

# Evaluation function
def evaluate(model, ldr_test, device):
    #ldr_test = DataLoader(dataset_test, batch_size = 256*4, shuffle = True)
    model.eval()
    with torch.no_grad():
        batch_acc = []
        for batch_idx, (images, labels) in enumerate(ldr_test):
            images, labels = images.to(device), labels.to(device)
                #---------forward prop-------------
            fx = model(images)
                
                # calculate accuracy
            acc = calculate_accuracy(fx, labels)
            batch_acc.append(acc.item())
        return sum(batch_acc)/len(batch_acc)
