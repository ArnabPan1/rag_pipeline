import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List
from transformers import AutoTokenizer, AutoModel
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import AutoModelForMaskedLM, AutoTokenizer

class EmbeddingModel:
    def __init__(self):
        self.dense_model = None
        self.dense_tokenizer = None
        self.doc_model = None
        self.doc_tokenizer = None
        self.query_model = None
        self.query_tokenizer = None
        self.load_tokenizer_and_model()
        
    def load_tokenizer_and_model(self):
        print("Loading tokenizer-----")
        dense_model, dense_tokenizer = self.load_dense_embedding_model()
        doc_model,doc_tokenizer, query_model, query_tokenizer = self.load_sparse_embedding_model()
        self.dense_model = dense_model
        self.dense_tokenizer = dense_tokenizer
        self.doc_model = doc_model
        self.doc_tokenizer = doc_tokenizer
        self.query_model = query_model
        self.query_tokenizer = query_tokenizer

    def load_dense_embedding_model(self):
        tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base")
        model = AutoModel.from_pretrained("thenlper/gte-base",
                                           #torch_dtype=torch.float16,
                                        attn_implementation="sdpa"
                                        ).to("cuda:0")
        return model,tokenizer
    
    def load_sparse_embedding_model(self):
        #document encoder
        doc_model_id = "naver/efficient-splade-VI-BT-large-doc"
        doc_tokenizer = AutoTokenizer.from_pretrained(doc_model_id)
        doc_model = AutoModelForMaskedLM.from_pretrained(doc_model_id).to("cuda:0")


        #query encoder
        query_model_id = "naver/efficient-splade-VI-BT-large-query"
        query_tokenizer = AutoTokenizer.from_pretrained(query_model_id)
        query_model = AutoModelForMaskedLM.from_pretrained(query_model_id).to("cuda:0")

        return doc_model,doc_tokenizer, query_model, query_tokenizer
    
    def average_pool(self,last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def get_embedding(self,
                      input_texts:List[str],
                      model:PreTrainedModel,
                      tokenizer:PreTrainedTokenizer):
        batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to("cuda:0")
        with torch.no_grad():
            outputs = model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.detach().cpu().tolist()
    
    def compute_sparse_vector(self,texts:str,
                                tokenizer, 
                                sparse_model):
        
        tokens = tokenizer(texts, return_tensors="pt",truncation=True).to("cuda:0")
        output = sparse_model(**tokens)
        logits, attention_mask = output.logits, tokens.attention_mask
        relu_log = torch.log(1 + torch.relu(logits)) #(seq_len,vocab_size) range -> (0,)
        weighted_log = relu_log * attention_mask.unsqueeze(-1)
        max_val, _ = torch.max(weighted_log, dim=1)
        vec = max_val.squeeze()
        
        torch.cuda.empty_cache()
        return vec, tokens