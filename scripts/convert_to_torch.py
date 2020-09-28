import os
import sys
import torch

allen = torch.load(os.path.join(sys.argv[1], 'model_state_epoch_19.th'), map_location=torch.device('cpu'))
out_dict = {".".join(k.split(".")[3:]): v for (k, v) in allen.items() if 'text_field_embedder' in k}
torch.save(out_dict, os.path.join(sys.argv[1], 'torch_model.pt'))
