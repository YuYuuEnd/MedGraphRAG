from GraphRAG import GraphRAG

sys = """
You are an assistant of a Mediacal professor.
Response wisely and concisely and as quick as possible.
""" 
# user = "What are the most effective strategies for managing progressive thoracic insufficiency in patients with fibrodysplasia ossificans progressiva(FOP)?"
user = "If you are an doctor, what instruction will you give to handle heart failure in baby situation."

# file_path = r"C:\Users\This PC\Downloads\02. Diagnosis and Treatment Manual author Patestos Dimitrios (1).pdf"
file_path = r"C:\Users\This PC\Downloads\Atlas of HEART FAILURE _ Cardiac Function and Dysfunction -- Arnold M_ Katz (auth_), Wilson S_ Colucci MD (eds_) -- Softcover reprint of the original -- 9781475745580 -- 5bec97274a4e43ffe879a28356cddf22 -- Anna’s.pdf"

rag_files_path = []
reference_files_path = []
vocab_files_path = []
target_path = "N1_"

rag = GraphRAG()
rag.Indexing(file_paths=[file_path], target_path = target_path, node_file='N1_nodes.json', edge_file='N1_edges.json')

rag.neo4j.clear_db()
rag.neo4j.add_node_to_db(file_path='N1_nodes.json', layer="A", type="Entity")
rag.neo4j.add_edge_to_db(file_path='N1_edges.json', layer="A", type="RELATIONSHIP")



# rag.upload_files_as_layer(rag_files_path, layer = "A", type = "EntityA")
# rag.upload_files_as_layer(reference_files_path, layer = "B", type = "EntityB")
# rag.upload_files_as_layer(vocab_files_path, layer = "C", type = "EntityC")

