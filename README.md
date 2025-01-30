# HELPFUL Vector Search System
A vector-based search and storing system using PostgreSQL. 

### embedding generation
- Uses PyTorch for embedding generation, which allows for far more flexibility when preparing the text for embedding and for pooling strategies which could be customized. 
- The vectors currently set to 768 dimensions.

Example output from `embedding_utils.py`:
```commandline
Generated embedding shape: (768,)
Generated 3 batch embeddings
```
This example is not a realistic use case, it is just for a demo. Instead of processing documents one a time, like I have been doing up to this point, the batch processing allows for multiple document processing simultaneously. Each item in a batch will be a different makerspace, with different `cache_keys` to help retrieve embeddings without recomputing.

### postgreSQL integration
- The system uses pgvector for vector storage and search.
- Tables are created for storing documents and their embeddings, with filtered search capabilities which can be improved significantly. 


### testing
```bash
# postgreSQL with vector extension
docker run -d \
  --name postgres-vector \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  ankane/pgvector
```
or, on windows powershell:
```powershell
docker run -d `
  --name postgres-vector `
  -e "POSTGRES_PASSWORD=postgres" `
  -p 5432:5432 `
  ankane/pgvector
```
then, create the vectordb database (pgSQL only creates a default postgres database):
```powershell
docker exec -it postgres-vector psql -U postgres
```
and now you can run the code:

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

to verify created databases, tables and their contents:

```bash
# get into the running PostgreSQL container
docker exec -it postgres-vector bash

psql -U postgres

#list all databases
\l
```
you should see vectordb in the list. 

to inspect the vectordb structure:

```bash
\c vectordb

#list all tables
\dt

#view table structure
\d makerspace_documents

# view first 10 entries
SELECT * FROM makerspace_documents LIMIT 10;

# check embeddings (first 3 dimensions)
SELECT id, title, chunk_type, embedding[1:3] AS partial_embedding 
FROM makerspace_documents 
LIMIT 5;
```
