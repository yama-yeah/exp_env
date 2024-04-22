import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from exp_env.data.data_maker_base import DataMaker, DataMakerSpec

class EmbeddingDataConfig(DataMakerSpec):
    model_name: str
    is_bardirectional: bool
    def __init__(self, model_name: str, is_bardirectional: bool,pad_token:str|None|bool=None,device:str='cpu'):
        self.model_name = model_name
        self.is_bardirectional = is_bardirectional
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if pad_token is True:
            self.tokenizer.pad_token=self.tokenizer.eos_token
        elif type(pad_token) is str:
            self.tokenizer.pad_token=pad_token
        model= AutoModel.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2').to(device)
        self.device=device
        self.emmbedding_weight = model.get_input_embeddings().weight.clone()

class EmbeddingDataMaker(DataMaker):
    def __init__(self, config: EmbeddingDataConfig):
        self.tokenizer = config.tokenizer
        self.embedding_weight = config.emmbedding_weight
    def make(self, texts:list,is_reverse:bool=True,max_length:int=256)->torch.Tensor:
        padding='right' if is_reverse else 'left'
        self.tokenizer.padding_side=padding
        #get embedding
        tokenized=self.tokenizer(texts, truncation=True, padding='max_length',max_length=max_length)
        ids=tokenized['input_ids']
        ids=torch.tensor(ids).to(self.embedding_weight.device)
        embedding= torch.nn.functional.embedding(ids,self.embedding_weight)
        return embedding,torch.tensor(tokenized['attention_mask']).to(embedding.device)

class EmbeddingsDataMaker(EmbeddingDataMaker):
    def __init__(self, config: EmbeddingDataConfig):
        self.is_berdirectional = config.is_bardirectional
        super().__init__(config)
    def make(self, texts:list[str],use_mask=False)->torch.Tensor:
        max_length=-1
        for text in texts:
            token=self.tokenizer(text, truncation=False, padding=False)['input_ids']
            if len(token)>max_length:
                max_length=len(token)
            del token
        forward,f_mask=super().make(texts,is_reverse=False,max_length=max_length)
        if self.is_berdirectional:
            reversed,_=super().make(texts,is_reverse=True,max_length=max_length)
            reversed=reversed.flip(1)

        if self.is_berdirectional:
            forward = torch.cat([forward,reversed],dim=2)
        if use_mask:
            return forward,f_mask
        return forward