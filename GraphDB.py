from neo4j import GraphDatabase
import json
from utils import *
from tqdm import tqdm
from data_loader import *

class GraphDBConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def execute_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]
        
    def clear_db(self):
        self.execute_query("MATCH (n) DETACH DELETE n")
        

    def add_node_to_db(self, file_path, layer, type, batch_size=64):
        nodes = load_from_json(file_path)
        
        query = """
        UNWIND $nodes AS node
        MERGE (n:""" + type + """ {id: node.id})
        SET n += node.properties
        SET n.layer = $layer
        """

        for i in tqdm(range(0, len(nodes), batch_size), desc="Uploading nodes"):
            batch = nodes[i:i + batch_size]
            formatted_batch = [{"id": node["id"], "properties": {k: v for k, v in node.items() if k != "id"}} for node in batch]
            try:
                self.execute_query(query, parameters={"nodes": formatted_batch, "layer": layer})
            except Exception as e:
                print(f"Error uploading batch {i // batch_size + 1}: {e}")   


    def add_edge_to_db(self, file_path, layer, type, batch_size=64):
        edges = load_from_json(file_path)

        query = """
        UNWIND $edges AS edge
        MATCH (n1:""" + type +""" {id: edge.id1}), (n2:""" + type +""" {id: edge.id2})
        MERGE (n1)-[r:RELATIONSHIP {type: edge.relationship}]->(n2)
        SET r.layer = $layer
        """

        for i in tqdm(range(0, len(edges), batch_size), desc="Uploading edges"):
            batch = edges[i:i + batch_size]
            try:
                self.execute_query(query, parameters={"edges": batch, "layer": layer})
            except Exception as e:
                print(f"Error uploading batch {i // batch_size + 1}: {e}")
 
    def retrieve_context(self, entities, layer):
        ret_query = """
        MATCH (n:EntityA)-[r:RELATIONSHIP]-(m:EntityA)
        WHERE n.layer = $layer AND m.layer = $layer
        WITH n, m, r
        WHERE id(n) < id(m)
        RETURN n, r, m
        """

        context = []
        try:
            res = self.execute_query(ret_query, {"layer": layer})
            
            new_entities = []
            max_loop = 10
            for i in range(len(res)):
                r = res[i]
                n = r['n']
                rel = r['r']
                m = r['m']

                if n['name'] in entities or m['name'] in entities:
                    context_str = (
                        f"{n['name']} {n['type']} {n['context']} "
                        f"{rel['type']} "
                        f"{m['name']} {m['type']} {m['context']}"
                    )
                    context.append(context_str)

                    if n['name'] not in entities:
                        new_entities.append(n['name'])
                    if m['name'] not in entities:
                        new_entities.append(m['name'])
                
                if i == len(res) - 1 and len(new_entities) > 0:
                    entities = new_entities
                    new_entities = []
                    i = 0
                    max_loop -= 1
                    if max_loop == 0:
                        break

        except Exception as e:
            print(f"Error retrieving context: {e}")

        return context
    
