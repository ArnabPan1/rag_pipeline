import os
import qdrant_client
from qdrant_client import models
from embedding_pipeline import EmbeddingModel
from chunking_strategy import ChunkingStrategy

from typing import List

class QuadRantPipe:
    def __init__(self):
        self.collection_name = "govt_land_data"
        self.emb_dimentions = 768
        self.client = None
        self.create_qudarant_config()
    
    
    def create_qdrant_collection(self,collection_name:str):
        self.client.create_collection(
        collection_name=collection_name,
        #dense vector index
        vectors_config={"text-dense":models.VectorParams(size=self.emb_dimentions, 
                                                        distance=qdrant_client.models.Distance.COSINE)},
        #sparse vector index
        sparse_vectors_config = {
            "text-sparse": models.SparseVectorParams(index=qdrant_client.models.SparseIndexParams(on_disk=False,))}
        )
        #get collection info 
        info = self.client.get_collection(collection_name=collection_name)
        print(f"created collection - {info}")
        return True
    
    def create_qudarant_config(self):
        self.client = qdrant_client.QdrantClient(":memory:")
        self.create_qdrant_collection(collection_name = self.collection_name)
        self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_id",
                field_schema="keyword",
                )
    def upload_documents(self, final_chunks, final_chunk_id,chunk_vectors_sparse,chunk_vectors_dense):
        summarys = [{"summary":chn} for chn in final_chunks]
        Points = [qdrant_client.models.PointStruct(id=final_chunk_id[i],
                    vector={"text-sparse": qdrant_client.models.SparseVector(
                        indices=chunk_vectors_sparse[i].get("indices").tolist(),

                    values=chunk_vectors_sparse[i].get("values").tolist()),
                    "text-dense":chunk_vectors_dense[i]},
                    payload=summarys[i]) 
          for i in range(len(final_chunks))
         ]
        self.client.upsert(collection_name=self.collection_name,
                points = Points)
        print("Uploading Done, Total No of chunks\n---")
        print(self.client.count(collection_name=self.collection_name))
    


class Pipeline:
    def __init__(self,text_file_path):
        self.chunk_strag = ChunkingStrategy(text_file_path)
        self.embed = EmbeddingModel()
        self.rag_index = QuadRantPipe()
        
    def get_chunk_dense_embedding(self):
        
        chunk_idxs = [idx for idx,doc in enumerate(self.final_chunks)]

        # get dense representation for each chunk.
        model = self.embed.dense_model
        tokenizer = self.embed.dense_tokenizer
        chunk_vectors_dense = [self.embed.get_embedding(input_texts=text,
                                            model=model,
                                            tokenizer=tokenizer) for text in self.final_chunks]
        

        return chunk_idxs, chunk_vectors_dense
    
    def get_chunk_sparse_embedding(self):
        chunk_vectors_sparse = []
        doc_tokenizer = self.embed.doc_tokenizer
        doc_model = self.embed.doc_model
        for chn in self.final_chunks:
            sparse_vec,_ = self.embed.compute_sparse_vector(chn,
                                                doc_tokenizer,
                                                doc_model)
            
            indices = sparse_vec.cpu().nonzero().numpy().flatten()
            sparse_vec_payload = {"indices":indices,
                                "values":sparse_vec.cpu().detach().numpy()[indices]}
            chunk_vectors_sparse.append(sparse_vec_payload)
        return chunk_vectors_sparse
    
    def upload_to_index(self):
        self.final_chunks = self.chunk_strag.start_chunking()
        chunk_idxs, chunk_vectors_dense = self.get_chunk_dense_embedding()
        chunk_vectors_sparse = self.get_chunk_sparse_embedding()
        self.rag_index.upload_documents(self.final_chunks, 
                                        chunk_idxs,
                                        chunk_vectors_sparse,
                                        chunk_vectors_dense)
    
        

if __name__ == "__main__":

    current_directory = os.getcwd()
    text_file_path = "result"
    text_file_path = os.path.join(current_directory, text_file_path)
    pp = Pipeline(text_file_path)
    pp.upload_to_index()

