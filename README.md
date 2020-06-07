# WIP
## Frame Semantic Identification
this is an **work in progress** porting from the original work at [open-sesame](https://github.com/swabhs/open-sesame) that was developed in dynet. 
Here I aim to re-implement dynet code to PyTorch 1.x. using python 3.x

As an additional, this model is implemented using Pytorch Lightning extension.

# Quickstart
## Frame Identification Model
### Model training
```
import torch
from model import FrameIdentificationRNN, FrameIdentificationParam
from conll09 import *

dev_conll_file_loc = 'data/fn1.7/fn1.7.dev.syntaxnet.conll'
# fill in vocabulary dictionary
e, m, x = read_conll(dev_conll_file_loc)

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

device = 'cuda' if torch.cuda.is_available() else 'cpu'

p = FrameIdentificationParam(**{
        'vocdict_size': len(VOCDICT._strtoint) + 2,
        'postdict_size': len(POSDICT._strtoint) + 2,
        'ludict_size': len(LUDICT._strtoint),
        'lpdict_size': len(LUPOSDICT._strtoint),
        'framedict_size': len(FRAMEDICT._strtoint),
        'batch_size': 32
    })
mdl = FrameIdentificationRNN(p, device=device)

logger = TensorBoardLogger('tb_logs', name='frame_id')
trainer = Trainer(
    gpus=1, 
    logger=logger,
    max_epochs=10
)
trainer.fit(mdl)
```
this model predict what frame will a sentence contains..

You can treat this model as PyTorch Model. 

### Inputs
All of these variable must be encoded as numbers and converted as tensor
  
|Variable|Description|Shape per Batch|
|---|---|---|
|Tokens| Sequence of tokens in a sentence|( batch_size, max_token_length)|
|Token Postags| Sequence of postag for each token in a sentence|(batch_size, max_token_length)|
|Lexical Unit| Lexical Unit of a token that will be classified as Frame|(batch_size, 1)|
|Lexical Unit Postag| Lexical Unit Postag  of a token that will be classified as Frame|(batch_size, 1)|
|Target Positions|Index of tokens in a sentence, where Frame element located|(batch_size, max_token_length)|

**Target positions must be tensor of 0 || 1, where 1 inidicating that the Frame Element will be located at that index** 

### Output
this model will produces logits, from input data.

## Frame Target Identification Model

### Model Training
```
import torch
from model import FrameTargetIdentificationRNN, FrameTargetIdentificationParam
from conll09 import *

dev_conll_file_loc = 'data/fn1.7/fn1.7.dev.syntaxnet.conll'
# fill in vocabulary dictionary
e, m, x = read_conll(dev_conll_file_loc)

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

device = 'cuda' if torch.cuda.is_available() else 'cpu'

p = FrameTargetIdentificationParam(**dict(
    input_size=len(VOCDICT._strtoint) + 2,
    postag_size=len(POSDICT._strtoint) + 2,
    lemma_size=len(LEMDICT._strtoint) + 2,
    output_size=87, #20
    pretrained_dim=300,
    batch_size=32
))
mdl = FrameTargetIdentificationRNN(p, device=device)

logger = TensorBoardLogger('tb_logs', name='frame_target_id')
trainer = Trainer(
    gpus=1, 
    logger=logger,
    max_epochs=10
)
trainer.fit(mdl)
```
### Inputs
All of these variable must be encoded as numbers and converted as tensor
  
|Variable|Description|Shape per Batch|
|---|---|---|
|Tokens| Sequence of tokens in a sentence|( batch_size, max_token_length)|
|Token Postags| Sequence of postag for each token in a sentence|(batch_size, max_token_length)|
|Lemma| Lexical Unit of a token that will be classified as Frame|(batch_size, max_token_length)| 

### Output
this model will produces logits, from input data.



