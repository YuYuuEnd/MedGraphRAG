from utils import *

default_model = "gemma3:1b"
def L_ent(user, model_name = default_model):
    # Extract entity from user query
    sys = """
    Extract all keywords related to MEDICAL field from the following text. For each keyword, provide:
    - Name: The name of the entity. (default: " ")
    - Type: The type of the entity (e.g. hormones, organs, symptom, methodology, medicine, condition, ...). (default: " ")
    - Context: A few sentences describing the entity based on provided context and your knowledge. (default:" ")
    Example for a entity:
    [
        {
            "name": "heart failure",
            "type": "symptom",
            "context": "the symptom that heart can not pump enough blood"
        },
    ]
    Response as quick as possible.
    Only response with the json array. If the array is empty just response [].
    """
    user = f"""
    Extract Medical keyword in following text and construct entity from them:
    {user}
    """
    response = call_llm(sys, user, model_name=model_name).lower()
    # print(response)
    return get_entity_list(response)

def L_sem(query1, query2, model_name = default_model):
    # Check if 2 query is semantically consistent (Not Done)
    sys = """
    You are being provided 2 context.
    You should determine if those 2 context maybe semantical related to each other.
    For example:
    Example 1:
    Context 1: "It is October 20th"
    Context 2: "year 2025"
    Answer: "yes"

    Example 2:
    Context 1: "Greg like pizza"
    Context 2: "Fox jumps over the dog"
    Answer: "no"
    """
    user = f"""
    Determine context 1 and context 2 have semantical related.
    Context 1:
    {query1}

    Context 2:
    {query2}
    """
    response = call_llm(sys, user, model_name=model_name)
    return response.lower().find("yes") != -1

def L_rel(query1, query2, model_name= default_model):
    sys = """
    You are being provided 2 query.
    Generate a concise sentence that describe the relationship of query 1 to query 2.
    Only generate the answer.
    For example:
    Query 1: "Heart failure is a condition that causes heart disease"

    Query 2: "Heart disease is a deadly disease"

    Answer: "Heart failure may lead to heart disease"
    """
    user = f"""
    Generate a concise sentences that describe the relationships of query 1 to query 2:
    Query 1:
    {query1}

    Query 2:
    {query2}
    
    Only response a sentence as fast as possible.
    """
    response = call_llm(sys, user, model_name=model_name)
    return response

def L_sim(query1, query2, model_name = "llama3.2:3b"):
    sys = """
    Assess the similarity of the two provided summaries and return a rating from these options: 'very similar', 'similar', 'general', 'not similar', 'totally not similar'. Provide only the rating.
    """
    user = f"""
    Query 1:
    {query1}

    Query 2:
    {query2}
    """
    rate = call_llm(sys, user, model_name=model_name)
    rate = rate.lower()
    if "totally not similar" in rate:
        return 0
    elif "not similar" in rate:
        return 1
    elif "general" in rate:
        return 2
    elif "very similar" in rate:
        return 4
    elif "similar" in rate:
        return 3
    else:
        print("llm returns no relevant rate")
        return -1
    
def L_res(sys, user, docs, model_name = "llama3.1:8b"):

    sys = """
    You are being provided documents related to the user question.
    Generate answer for user question based on documents provided.
    Keep the answer wisely and concisely.
    """
    user = f"""
    Documents:
    {docs}

    Question:
    {user}

    Answer:
    """
    response = call_llm(sys, user, model_name)
    return response
