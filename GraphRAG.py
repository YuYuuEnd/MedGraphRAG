from data_loader import *
from LGs import *
from tqdm import tqdm
from utils import *

from alive_progress import alive_bar
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

from GraphDB import GraphDBConnection
import random
import os
num_workers = os.cpu_count() -1

class GraphRAG:

    nodes_path = "nodes1.json"
    edges_path = "edges1.json"
    
    def __init__(self):
        self.neo4j = GraphDBConnection(
                uri      = "neo4j+s://cc94cc74.databases.neo4j.io",
                user     = "neo4j",
                password = "cD4ALjvgZ-uhvbQHs2dpYOCHVMQwws_TMcqENgnIKcw"
                )

    """
    Index Part
    """
    def Indexing(self, file_paths, target_path, chunk_size=200, node_file=None, edge_file=None):
        if node_file is None:
            data = []
            data = self.Chunking(file_paths, chunk_size)

            entities = self.Extract_Entity(data)
            load_to_json(target_path+"nodes.json", entities)
        else:
            self.nodes_path = node_file
            entities = load_from_json(node_file)

        if edge_file is None:
            r_entities = random.sample(entities, 1000)
            relationships = self.Relationship_Linking(r_entities)
            load_to_json(target_path+"edges.json", relationships)
        else:
            self.edges_path = edge_file
            relationships = load_from_json(edge_file)


    def Chunking(self, file_paths, chunk_size=200, num_workers=num_workers):
        def process_file(file_path):
            return data_load(file_path, chunk_size=chunk_size, chunk_overlap=0)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(process_file, file_paths), total=len(file_paths), desc="Chunking"))

        data = [item for sublist in results for item in sublist]
        data = self.semantic_chunking(data)

        return data
    
    def semantic_chunking(self, data):
        return data

    def Extract_Entity(self, data, num_workers=4):
        def process_chunk(chunk):
            entities =  L_ent(chunk.page_content, model_name="stablelm2")
            # return filtering_node(entities)
            return entities
        
        entities = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(process_chunk, data), total=len(data), desc="Extract Entity"))
        for result in results:
            entities.extend(result)

        
        entities = filtering_node(entities)
        entities = self.Generate_Embeddings(entities=entities)
        merge_similar_nodes(entities)

        for entity in entities:
            entity["id"] = get_id()
        return entities
    
    def Generate_Embeddings(self, entities, num_workers=num_workers, batch_size=32):
        def process_batch(batch):
            contexts = [entity["context"] for entity in batch]
            embeddings = get_emb(contexts)  # Batch embedding generation
            for entity, emb in zip(batch, embeddings):
                entity["emb"] = emb.tolist()  # Convert NumPy array to list
            return batch

        batches = [entities[i:i + batch_size] for i in range(0, len(entities), batch_size)]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(process_batch, batches), total=len(batches), desc="Generate Embeddings"))
        
        entities = [entity for batch in results for entity in batch]

        return entities

    def Relationship_Linking(self, entities, threshold=0.5, num_workers=num_workers, drop_rate=0.2, batch_size=100):
        def process_batch(batch):
            results = []
            for pair in batch:
                if random.random() < drop_rate:
                    continue
                e1, e2 = (entities[pair[0]], entities[pair[1]])
                if cosine_similarity(e1["emb"], e2["emb"]) > threshold:
                    results.append({
                        "id1": e1["id"],
                        "id2": e2["id"],
                        "relationship": L_rel(
                            e1["name"] + " " + e1["type"] + " " + e1["context"],
                            e2["name"] + " " + e2["type"] + " " + e2["context"],
                            model_name="gemma3:1b"
                        )
                    })
            return results

        print("Pairing entities...")
        entity_pairs = list(combinations(range(len(entities)), 2))  # Unique pairs
        print(f"Total pairs: {len(entity_pairs)}")

        # entity_pairs = [pair for pair in entity_pairs if cosine_similarity(entities[pair[0]]["emb"], entities[pair[1]]["emb"]) > threshold]
        # Split pairs into batches
        batches = [entity_pairs[i:i + batch_size] for i in range(0, len(entity_pairs), batch_size)]

        relationships = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            with alive_bar(len(batches), title="Relationship Linking") as bar:
                for batch_results in executor.map(process_batch, batches):
                    relationships.extend(batch_results)
                    bar()  # Update the progress bar
        return relationships
    

    # Upload to DB
    # def clear_db(self):
    #     self.neo4j.clear_db()

    # def upload_nodes_to_neo4j(self, file_path, layer, type):
    #     self.neo4j.add_node_to_db(file_path=file_path, layer=layer, type=type)

    # def upload_edges_to_neo4j(self, file_path, layer, type):
    #     self.neo4j.add_edge_to_db(file_path=file_path, layer=layer, type=type)

    # def upload_file_as_layer(self, file_part, layer, type):
        

    """
    Retrieve Part
    """
    def Retrieve(self, query, k=5, model_name = "llama3.1:8b"):
        entities = L_ent(query, model_name=model_name)
        entities_name = []
        for entity in entities:
            entities_name.append(entity["name"])

        contexts = self.neo4j.retrieve_context(entities_name, layer = "A")
        rating_list = self.rating_context(query, contexts)

        rating_list.sort(key=lambda x: x[1], reverse=True)
        top_k_contexts = [item[0] for item in rating_list[:k]]
        return top_k_contexts

    def rating_context(self, query, contexts, num_workers=num_workers):
        def rate_context(context):
            return (context, L_sim(context, query))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            rating_list = list(tqdm(executor.map(rate_context, contexts), total=len(contexts), desc="Rating Context"))

        return rating_list

    """
    Generate Part
    """
    def Generate(self, sys, user):

        response = L_res(sys, user, "", model_name="llama3.2:3b")
        query = f"""
        Draft: {response}
        Question: {user}
        """
        print("Retrieve")
        related_doc = self.Retrieve(query, k=4)
        docs = "\n\n".join(doc for doc in related_doc)
        docs = "DOCS:" + docs + "\n\n"
        response = "RESPONSE:" + L_res(sys, user, docs, model_name="llama3.1:8b")
        
        load_to_json("debug.json", docs + response)
        return response