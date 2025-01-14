# HELPFUL Vector Search System
A vector-based search and storing system using PostgreSQL. 

### embedding generation
- Uses PyTorch with a GTE base model for embedding generation. 
- It allows for far more flexibility when preparing the text for embedding and for pooling strategies which could be adjusted for our use case. 
- The vectors representing semantic content are currently set to 768 dimensions.

Example output from `embedding_utils.py`:
```commandline
Generated embedding shape: (768,)
Generated 3 batch embeddings
```
This example is not a realistic use case, it is just for a demo. Instead of processing documents one a time, like I have been doing up to this point, the batch processing allows for multiple document processing simultaneously. Each item in a batch will be a different makerspace, with different `cache_keys` to help retrieve embeddings without recomputing.

### postgreSQL integration
- The system uses pgvector for vector storage and search.
- Currently using cosine distance.
- Tables are created for storing documents and their embeddings, with filtered search capabilities which can be improved significantly. 

**Need to create a Dockerfile and docker-compose for easier deployment and testing**

### testing
```bash
# postgreSQL with vector extension
docker run -d \
  --name postgres-vector \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  ankane/pgvector
```
```python
# Initialize database
vector_db = MakerspaceVectorDB(db_params)
vector_db.initialize_db()

# Store documents
documents = load_and_process_documents('./OKWs/')
vector_db.store_documents(documents)

# Search example
results = vector_db.search(
    "Looking for metal fabrication capabilities",
    limit=5,
    filters={'chunk_type': 'inventory-atoms'}
)
```