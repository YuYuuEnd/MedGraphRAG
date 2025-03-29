from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.embeddings import sentence_transformers_embeddings as ste

from langchain_ollama import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings

from sentence_transformers import SentenceTransformer

from tqdm import tqdm
import numpy as np
import uuid
import json
import torch

def call_llms(sys, user, response_format = None):
    ollama_model = ModelFactory.create(
        model_platform    = ModelPlatformType.OLLAMA,
        model_type        = "llama3.1:8b",
        # url               = "http://localhost:11434/v1",
        model_config_dict = {"temperature": 0.5}
        )
    sys = f"""
    You are a helpful assistant of a medical professor.
    Response as fast as possible. Ideal time is under 0.5 seconds.
    {sys}
    """
    agent = ChatAgent(sys, model = ollama_model, token_limit = 500)
    return agent.step(user, response_format = None).msg.content

def call_llm(sys, user, model_name = "llama3.1:8b"):
    llm = OllamaLLM(
        model = model_name,
        # system = sys,
        )
    return llm.invoke(sys + user)


def get_emb(query):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    if isinstance(query, str):
        # Single query
        return embedding_model.encode(query, convert_to_numpy=True)
    elif isinstance(query, list):
        # Batch processing for multiple queries
        return embedding_model.encode(query, convert_to_numpy=True)
    else:
        raise ValueError("Query must be a string or a list of strings.")

def get_id():
    # Return random string used as id
    return str(uuid.uuid4())[:10]

def get_entity_list(nodes:str):
    start = nodes.find("[")
    end = nodes.find("]")
    if start == -1 or end == -1:
        return []
    json_string = nodes[start: end + 1]
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return []



def cosine_similarity(vec1, vec2):
    vec1 = torch.tensor(vec1, device="cuda")
    vec2 = torch.tensor(vec2, device="cuda")
    similarity = torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))
    return similarity.item()

def cosine_similarity_cpu(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def merge_similar_nodes(nodes, threshold = 0.75):
    new_entities = []
    for i in tqdm(range(len(nodes)), desc="Merging Node"):
        entity = nodes[i]
        is_merged = False
        for new_entity in new_entities:
            if (cosine_similarity_cpu(entity["emb"], new_entity["emb"]) > threshold):
                is_merged = True
        if not is_merged:
            new_entities.append(entity)
    print("n Nodes:", len(nodes), "->", len(new_entities))
    return new_entities


def has_enough_properties(entity, required_properties=None):
    if required_properties is None:
        required_properties = ["name", "type", "context"]

    # Check if the entity is a dictionary
    if not isinstance(entity, dict):
        print(f"Anomaly detected: Entity is not a dictionary: {entity}")
        return False

    for key in required_properties:
        # Check if the key exists in the entity
        if key not in entity:
            print(f"Anomaly detected: Missing key '{key}' in entity {entity}")
            return False

        # Check if the value is valid (not None, not empty, and correct type)
        value = entity[key]
        if value is None or (isinstance(value, str) and value.strip() == ""):
            print(f"Anomaly detected: Invalid value for key '{key}' in entity {entity}")
            return False

    return True


def filtering_node(nodes):
    new_entities = []
    for i in tqdm(range(len(nodes)), desc="Filtering Node"):
        entity = nodes[i]
        if not has_enough_properties(entity):
            continue
        new_entities.append(entity)
    return new_entities

