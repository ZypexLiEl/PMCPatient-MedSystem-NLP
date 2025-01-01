from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, set_seed, logging
from torch.utils.data import DataLoader
from torch.nn import Linear, Module
from typing import Dict, List
from collections import Counter, defaultdict
from itertools import chain
import torch

logging.set_verbosity_error()

torch.manual_seed(0)
set_seed(34)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

class MimicTransformer(Module):
    def __init__(self, tokenizer_name, num_labels=738, cutoff=512, model_path=None):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.num_labels = num_labels
        self.cutoff = cutoff

        # Initialize config, tokenizer, and model
        self.config = AutoConfig.from_pretrained(self.tokenizer_name, num_labels=self.num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", cache_dir=r"D:/transformers/Bio_ClinicalBERT/tokenizer")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.tokenizer_name, config=self.config)

        # Set model to evaluation mode
        self.model.eval()

        # Additional linear layer
        self.linear = Linear(in_features=self.cutoff, out_features=1)

        # Load pre-trained weights if a path is provided
        if model_path:
            self._load_pretrained_weights(model_path)

    def _load_pretrained_weights(self, path):
        """Loads pre-trained weights into the model."""
        state_dict = torch.load(path, map_location='cpu')
        self.model.load_state_dict({"model." + k: v for k, v in state_dict.items()}, strict=False)
    
    def parse_icds(self, instances: List[Dict]):
        token_list = defaultdict(set)
        token_freq_list = []
        for instance in instances:
            icds = list(chain(*instance['icd']))
            icd_dict_list = list({icd['start']: icd for icd in icds}.values())
            for icd_dict in icd_dict_list:
                icd_ent = icd_dict['text']
                icd_tokenized = self.tokenizer(icd_ent, add_special_tokens=False)['input_ids']
                icd_dict['tokens'] = icd_tokenized
                icd_dict['labels'] = []
                for i,token in enumerate(icd_tokenized):
                    if i != 0:
                        label = "I-ATTN"
                    else:
                        label = "B-ATTN"
                    icd_dict['labels'].append(label)
                    token_list[token].add(label)
                    token_freq_list.append(str(token) + ": " + label)
        token_tag_freqs = Counter(token_freq_list)
        for token in token_list:
            if len(token_list[token]) == 2:
                inside_count = token_tag_freqs[str(token) + ": I-ATTN"]
                begin_count = token_tag_freqs[str(token) + ": B-ATTN"]
                if begin_count > inside_count:
                    token_list[token].remove('I-ATTN')
                else:
                    token_list[token].remove('B-ATTN')
        return token_list
    

    def collate_mimic(
            self, instances: List[Dict], device='cuda'
    ):
        tokenized = [
            self.tokenizer.encode(
                ' '.join(instance['description']), max_length=self.cutoff, truncation=True, padding='max_length'
            ) for instance in instances
        ]
        entries = [instance['entry'] for instance in instances]
        labels = torch.tensor([x['drg'] for x in instances], dtype=torch.long).to(device).unsqueeze(1)
        inputs = torch.tensor(tokenized, dtype=torch.long).to(device)
        icds = self.parse_icds(instances)
        xai_labels = torch.zeros(size=inputs.shape, dtype=torch.float32).to(device)
        for i,row in enumerate(inputs):
            for j,ele in enumerate(row):
                if ele.item() in icds:
                    xai_labels[i][j] = 1
        return {
            'text': inputs,
            'drg': labels,
            'entry': entries,
            'icds': icds,
            'xai': xai_labels
        }

    def forward(self, input_ids, attention_mask=None, drg_labels=None):
        if drg_labels:
            cls_results = self.model(input_ids, attention_mask=attention_mask, labels=drg_labels, output_attentions=True)
        else:
            cls_results = self.model(input_ids, attention_mask=attention_mask, output_attentions=True)
        last_attn = cls_results[-1][-1] # (batch, attn_heads, tokens, tokens)
        # last_attn = torch.mean(torch.stack(cls_results[-1])[:], dim=0)
        # last_layer_attn = torch.mean(last_attn[:, :-3, :, :], dim=1)
        last_layer_attn = last_attn[:, -1, :, :]
        xai_logits = self.linear(last_layer_attn).squeeze(dim=-1)
        return (cls_results, xai_logits)
    
    def find_tokenizer(self, tokenizer_name):
        """
    
        :param args:
        :return:
        """
        if tokenizer_name == 'clinical_longformer':
            return 'yikuan8/Clinical-Longformer'
        if tokenizer_name == 'clinical':
            return 'emilyalsentzer/Bio_ClinicalBERT'
        else:
            # standard transformer
            return 'bert-based-uncased'      