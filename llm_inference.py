from transformers import AutoTokenizer
import qdrant_client
from qdrant_client import models
from vllm import LLM, SamplingParams
from prompts import system_prompt
from embedding_pipeline import EmbeddingModel

class VllmQwen:
    def __init__(self):
        self.collection_name = "govt_land_data"
        self.client = qdrant_client.QdrantClient(":memory:")
        self.embed = EmbeddingModel()
        self.llm_tok, self.llm_model = self.load_qwen_model()

    def load_qwen_model(self):
        llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct",
                dtype="half")

        return llm_tokenizer,llm
    
    def generate_text(self,system_prompt,user_prompt,llm_tokenizer,llm):
        all_text = []
        sampling_params = SamplingParams(temperature=0.7, 
                                         top_p=0.8, 
                                         repetition_penalty=1.05, 
                                         max_tokens=512)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        text = llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        outputs = llm.generate([text], sampling_params)
        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"Generated text: {generated_text!r}")
            print("__________")
            all_text.append(generated_text)
        return all_text
    
    def get_query_vector(self,
                    query,
                     dense_model,
                     sparse_model,
                     dense_tokenizer,
                     sparse_tokenizer,
                     )->dict:
    
        #query sparse vector
        sparse_vec,_ = self.embed.compute_sparse_vector(query,
                                            sparse_tokenizer,
                                            sparse_model)

        sparse_indices = sparse_vec.cpu().nonzero().numpy().flatten()
        sparse_scores = sparse_vec.cpu().detach().numpy()[sparse_indices]
        
        dense_vector = self.embed.get_embedding(input_texts=[query,],
                                            model=dense_model,
                                            tokenizer=dense_tokenizer)[0]
        
        return {"sparse_vector":{"indices":sparse_indices,
                                "values":sparse_scores},
                "dense_vector":dense_vector}
    
    def hybrid_vector_search(self,
                             query:str,
                            filters=None,
                            total_results = 10,
                            ):
        client = self.client
        query_vector = self.get_query_vector(query,
                                             self.embed.dense_model,
                                            self.embed.query_model,
                                            self.embed.dense_tokenizer,
                                            self.embed.query_tokenizer)
        
        prefetch_n_sparse = total_results*4
        prefetch_n_dense = total_results*4
        
        prefetch=[
            models.Prefetch(
                query=models.SparseVector(**query_vector.get("sparse_vector")),
                using="text-sparse",
                limit=prefetch_n_sparse,
                filter=filters
            ),
            models.Prefetch(
                query=query_vector.get("dense_vector"),  
                using="text-dense",
                limit=prefetch_n_dense,
                filter=filters
            ),
        ]
        
        results = client.query_points(
            self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),
            with_payload=True,
            limit=total_results,
            )
        return results
    
    def retriver_pipeline(self,user_query_list):
        all_response = []
        for query in user_query_list:
            all_context = []
            for obj in self.hybrid_vector_search(query,total_results=2).points:
                print(obj,end="\n\n")
                all_context.append(obj.payload.get('summary',''))
            user_prompt = all_context
            res_text = self.generate_text(system_prompt,
                                        user_prompt,
                                        self.llm_tok,
                                        self.llm_model)
            all_response.append(res_text)
        return all_response


if __name__ == "__main__":
    user_query_list = ["who is the Managing Partner of JB infra group",
                       "where does rammohan stay?",
                       "who is the holder of attorny doct no 5606/2016"]
    pp = VllmQwen()
    generated_answer = pp.retriver_pipeline(user_query_list)



