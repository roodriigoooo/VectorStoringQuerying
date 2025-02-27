import psycopg2
from psycopg2.extras import execute_values
from embedding_utils import SupplyChainEmbedder
from typing import List, Dict, Any
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MakerspaceVectorDB:
    def __init__(self, db_params):
        self.db_params = db_params
        self.embedder = SupplyChainEmbedder()

    def initialize_db(self):
        """Initialize database, create if it does not exist, then create tables and extensions"""
        try:
            temp_db_params = self.db_params.copy()
            temp_db_params["dbname"] = "postgres"
            conn = psycopg2.connect(**temp_db_params)
            conn.autocommit = True # CREATE DATABASE can't run in a transaction
            cur = conn.cursor()

            db_name = self.db_params["dbname"]
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (db_name,))
            exists = cur.fetchone()

            if not exists:
                cur.execute(f"CREATE DATABASE {db_name};")
                logger.info(f"Created database {db_name}")

            cur.close()
            conn.close()

        except Exception as e:
            logger.error(f"Error ensuring database exists: {e}")
            raise

        conn = psycopg2.connect(**self.db_params)
        cur = conn.cursor()

        try:
            # enable vector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # create makerspace_documents table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS makerspace_documents (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    chunk_type TEXT,
                    embedding vector(768),
                    metadata JSONB
                );
            """)

            # create index for vector similarity search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS makerspace_embedding_idx 
                ON makerspace_documents 
                USING ivfflat (embedding vector_cosine_ops);
            """)

            conn.commit()
            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()

    def store_documents(self, documents):
        """Store documents with their embeddings"""
        conn = psycopg2.connect(**self.db_params)
        cur = conn.cursor()

        try:
            # process documents in batches
            batch_size = 8
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]

                # prepare batch data
                batch_data = []
                for doc in batch:
                    embedding = self.embedder.generate_embedding(
                        doc.page_content,
                        cache_key=f"{doc.metadata['title']}_{doc.metadata.get('chunk_type', 'unknown')}"
                    )

                    # debugging
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()

                    batch_data.append((
                        doc.metadata.get('title', ''),
                        doc.page_content,
                        doc.metadata.get('chunk_type', 'unknown'),
                        embedding,
                        json.dumps(doc.metadata)
                    ))

                # insert batch
                execute_values(
                    cur,
                    """
                    INSERT INTO makerspace_documents 
                    (title, content, chunk_type, embedding, metadata)
                    VALUES %s
                    """,
                    batch_data,
                    template="(%s, %s, %s, %s::vector, %s)"
                )

                conn.commit()
                logger.info(f"Processed batch of {len(batch)} documents")

            logger.info(f"Successfully stored {len(documents)} documents")

        except Exception as e:
            logger.error(f"Error storing documents: {e}")
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()

    def search(self, query, limit = 5, filters = None):
        '''Search for similar documents'''
        conn = psycopg2.connect(**self.db_params)
        cur = conn.cursor()

        try:
            # Generate query embedding
            query_embedding = self.embedder.generate_embedding(query)

            #debugging
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            # Build query
            query_parts = [
                "SELECT title, content, chunk_type, 1 - (embedding <=> %s::vector) AS similarity",
                "FROM makerspace_documents"
            ]
            params = [query_embedding]

            # add filters if provided
            if filters:
                conditions = []
                if 'chunk_type' in filters:
                    conditions.append("chunk_type = %s")
                    params.append(filters['chunk_type'])
                if 'title' in filters:
                    conditions.append("title ILIKE %s")
                    params.append(f"%{filters['title']}%")
                if conditions:
                    query_parts.append("WHERE " + " AND ".join(conditions))

            # add ordering and limit
            query_parts.extend([
                "ORDER BY similarity DESC",
                "LIMIT %s"
            ])
            params.append(limit)

            # execute search
            cur.execute(" ".join(query_parts), params)
            results = cur.fetchall()

            # format results
            formatted_results = []
            for row in results:
                formatted_results.append({
                    'title': row[0],
                    'content': row[1],
                    'chunk_type': row[2],
                    'similarity': row[3]
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Error performing search: {e}")
            raise
        finally:
            cur.close()
            conn.close()


# Example usage
if __name__ == "__main__":
    from preprocessing import load_and_process_documents

    # Database configuration
    db_params = {
        "dbname": "vectordb",
        "user": "postgres",
        "password": "postgres",
        "host": "localhost",
        "port": "5432"
    }

    # Initialize vector database
    vector_db = MakerspaceVectorDB(db_params)
    vector_db.initialize_db()

    # Load and store documents
    directory_path = './OKWs/'
    documents = load_and_process_documents(directory_path)
    vector_db.store_documents(documents)

    # Example search
    query = "woodworking and fabric capabilities"
    results = vector_db.search(
        query,
        limit=5,
        filters={'chunk_type': 'inventory-atoms'}
    )

    print(f"\nQuery: {query}")
    print("\nMatching makerspaces:")
    for result in results:
        print(f"\nTitle: {result['title']}")
        print(f"Type: {result['chunk_type']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Content: {result['content'][:200]}...")