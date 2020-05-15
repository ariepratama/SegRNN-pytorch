# WIP
## Frame Semantic Identification
this is an **work in progress** porting from the original work at [open-sesame](https://github.com/swabhs/open-sesame) that was developed in dynet. 
Here I aim to re-implement dynet code to PyTorch 1.x. using python 3.x

## How To Use
### 1. Load Libraries
```python
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm
from model import FrameIdentificationRNN, Param, configuration
from conll09 import *
```
### 2. Instantiate Model
```python
p = Param(**{
        'vocdict_size': len(VOCDICT._strtoint),
        'postdict_size': len(POSDICT._strtoint),
        'ludict_size': len(LUDICT._strtoint),
        'lpdict_size': len(LUPOSDICT._strtoint),
        'framedict_size': len(FRAMEDICT._strtoint)
    })
mdl = FrameIdentificationRNN(None, p).to(device)
```

### 3. Define Training Routine
```python
optimizer = optim.Adam(mdl.parameters(), lr=0.001)
criterion = nn.NLLLoss()

def train_one_epoch(mdl: nn.Module, optimizer: optim.Optimizer, criterion, targets: list, inputs: list, device: str):
    running_loss = 0
    for target, inp in tqdm(zip(targets, inputs)):
        optimizer.zero_grad()

        y_pred_val, y_pred = torch.topk(mdl(*inp, device), 1)
        
        loss = criterion(y_pred_val.squeeze().view(1, -1), target.unsqueeze(0))

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    return float(running_loss) / len(inputs)

def train(mdl: nn.Module, epoch: int, inputs: list, targets: list, optimizer: optim.Optimizer, criterion, device: str):
    for i in range(epoch):
        print('epoch: {}'.format(i))
        epoch_loss = train_one_epoch(mdl, optimizer, criterion, targets, inputs, device)
        print('epoch loss: {}'.format(epoch_loss))
```

### 4. Then run the training routine
```python
epoch = 2
inputs = [
    (
        torch.tensor(conll_example.tokens).to(device),
        torch.tensor(conll_example.postags).to(device),
        conll_example.lu,
        list(conll_example.targetframedict.keys())
    )
    for conll_example in e
]

targets = [
    torch.tensor(conll_example.frame.id).to(device)
    for conll_example in e
]

train(mdl, epoch, inputs, targets, optimizer, criterion, device)

```