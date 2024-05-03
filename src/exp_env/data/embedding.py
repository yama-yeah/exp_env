import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA, FastICA

from exp_env.data.data_maker_base import DataMaker, DataMakerSpec

class ReduceWeightDimConfig:
    def __init__(self, mode: str, dim: int):
        self.mode = mode
        self.dim = dim
    @staticmethod
    def pca(dim: int):
        return ReduceWeightDimConfig('pca', dim)
    @staticmethod
    def tsne(dim: int):
        return ReduceWeightDimConfig('tsne', dim)
    @staticmethod
    def umap(dim: int):
        return ReduceWeightDimConfig('umap', dim)
    @staticmethod
    def ica(dim: int):
        return ReduceWeightDimConfig('ica', dim)
    @staticmethod
    def none():
        return ReduceWeightDimConfig('none', 0)

class EmbeddingDataConfig(DataMakerSpec):
    model_name: str
    is_bardirectional: bool
    def __init__(self, model_name: str, is_bardirectional: bool,pad_token:str|None|bool=None,time_shift=0,alpha=0.3,device:str='cpu',reduce_weight_dim: ReduceWeightDimConfig=ReduceWeightDimConfig.none(),quantization_config=None):
        self.model_name = model_name
        self.is_bardirectional = is_bardirectional
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if pad_token is True:
            self.tokenizer.pad_token=self.tokenizer.eos_token
        elif type(pad_token) is str:
            self.tokenizer.pad_token=pad_token
        model= AutoModel.from_pretrained(model_name,
            quantization_config=quantization_config
            )
        self.device=device
        self.emmbedding_weight = model.get_input_embeddings().weight.cpu().detach().numpy()
        if reduce_weight_dim.mode=='pca':
            if reduce_weight_dim.dim<0:
                #auto select dim
                reduce_weight_dim.dim=self.emmbedding_weight.shape[1]
                pca=PCA(n_components=reduce_weight_dim.dim)
                #寄与度が0.9以上になる次元数を選択
                pca.fit(self.emmbedding_weight)
                for i in range(reduce_weight_dim.dim):
                    if np.sum(pca.explained_variance_ratio_[:i])>0.9:
                        reduce_weight_dim.dim=i
                        break
                del pca
            pca=PCA(n_components=reduce_weight_dim.dim)
            self.emmbedding_weight=pca.fit_transform(self.emmbedding_weight)
            del pca
        elif reduce_weight_dim.mode=='ica':
            ica=FastICA(n_components=reduce_weight_dim.dim)
            self.emmbedding_weight=ica.fit_transform(self.emmbedding_weight)
            del ica
        self.emmbedding_weight=torch.tensor(self.emmbedding_weight).to(device)
        self.emmbedding_weight.requires_grad=False
        self.emmbedding_out_dim=reduce_weight_dim.dim
        self.time_shift=time_shift
        self.alpha=alpha

class EmbeddingDataMaker(DataMaker):
    def __init__(self, config: EmbeddingDataConfig):
        self.tokenizer = config.tokenizer
        self.embedding_weight = config.emmbedding_weight
    def make(self, texts:list[list[int]]|list[str],is_reverse:bool=True,max_length:int=256)->torch.Tensor:
        padding='right' if is_reverse else 'left'
        self.tokenizer.padding_side=padding
        #get embedding
        if type(texts[0][0]) is int:
            tokenized={}
            tokenized['input_ids']=[text+[self.tokenizer.pad_token_id]*(max_length-len(text))for text in texts]
            tokenized['attention_mask']=[[1]*len(text)+[0]*(max_length-len(text))for text in texts]
        else:
            tokenized=self.tokenizer(texts, truncation=True, padding='max_length',max_length=max_length)
        ids=tokenized['input_ids']
        ids=torch.tensor(ids).to(self.embedding_weight.device)
        embedding= torch.nn.functional.embedding(ids,self.embedding_weight)
        #masking
        embedding=embedding*torch.tensor(tokenized['attention_mask']).unsqueeze(2).to(embedding.device)
        
        return embedding,torch.tensor(tokenized['attention_mask']).to(embedding.device)

class EmbeddingsDataMaker(EmbeddingDataMaker):
    def __init__(self, config: EmbeddingDataConfig):
        self.is_berdirectional = config.is_bardirectional
        self.time_shift=config.time_shift
        self.alpha=config.alpha
        super().__init__(config)
    def make(self, texts:list[str]|list[list[int]],use_mask=False,is_reverse=False)->torch.Tensor:
        max_length=-1
        time_shift=self.time_shift
        alpha=self.alpha
        for text in texts:
            if type(text) is str:
                token=self.tokenizer(text, truncation=False, padding=False)['input_ids']
            else:
                token=text
            if len(token)>max_length:
                max_length=len(token)
            del token
        forward,f_mask=super().make(texts,is_reverse=is_reverse,max_length=max_length)
        if self.is_berdirectional:
            reversed,_=super().make(texts,is_reverse=not is_reverse,max_length=max_length)
            reversed=reversed.flip(1)

        if self.is_berdirectional:
            forward = torch.cat([forward,reversed],dim=2)
        if time_shift>0:
            forward=alpha*forward+(1-alpha)*torch.roll(forward,time_shift,dims=1)
        if use_mask:
            return forward,f_mask
        return forward