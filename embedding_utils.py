import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import json
import numpy as np


class SupplyChainEmbedder:
    def __init__(self):
        # init model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('thenlper/gte-base')
        self.model = AutoModel.from_pretrained('thenlper/gte-base')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        # cache for embeddings
        self._embedding_cache = {}
        self.max_length = 512
        self.normalize_embeddings = True

    def _prepare_text_for_embedding(self, data):
        '''Prepare text from JSON data with supply chain focus'''
        text_parts = []

        # core information with weights, for example. this could be improved a lot, probably.
        text_parts.extend([
            data.get('title', '') * 2,  # Repeat title for emphasis
            data.get('description', '')
        ])

        # keywords with domain-specific emphasis, for example
        keywords = data.get('keywords', [])
        supply_chain_keywords = [k for k in keywords if any(term in k.lower()
            for term in ['supply', 'chain', 'logistics', 'inventory', 'production'])]
        text_parts.extend(supply_chain_keywords * 2)  # Emphasize supply chain related keywords
        text_parts.extend(keywords)

        # process atoms with supply chain context, all this to show how it can be customized to our use case
        for atom_type in ['inventory-atoms', 'product-atoms', 'tool-list-atoms']:
            atoms = data.get(atom_type, [])
            for atom in atoms:
                # emphasize supply chain relevant information
                text_parts.append(f"{atom.get('identifier', '')} {atom.get('description', '')}")
                if 'capacity' in atom or 'lead_time' in atom or 'availability' in atom:
                    text_parts.append(atom.get('description', '') * 2)  # Emphasize supply chain metrics

        # add process information
        processes = data.get('processes', [])
        for process in processes:
            text_parts.append(f"{process.get('identifier', '')} {process.get('description', '')}")

        return ' '.join(text_parts)

    def _average_pooling(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        '''average pooling operation on token embeddings'''
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    @torch.no_grad()
    def generate_embedding(self, data, cache_key = None) -> np.ndarray:
        '''Generate embeddings with caching support'''
        if cache_key:
            if cache_key in self._embedding_cache:
                return self._embedding_cache[cache_key]

        # prepare text
        if isinstance(data, dict):
            text = self._prepare_text_for_embedding(data)
        else:
            text = str(data)

        # tokenize
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # generate embeddings
        outputs = self.model(**inputs)
        embeddings = self._average_pooling(outputs.last_hidden_state, inputs['attention_mask'])

        # normalize if configured
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # convert to numpy and cache
        embeddings_np = embeddings.detach().cpu().numpy()[0]
        if cache_key:
            self._embedding_cache[cache_key] = embeddings_np

        return embeddings_np

    def batch_generate_embeddings(self, data_list, cache_keys = None, batch_size = 8):
        '''Generate embeddings in batches for efficiency'''
        embeddings = []
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            batch_keys = cache_keys[i:i + batch_size] if cache_keys else None

            batch_to_process = []
            batch_indices = []
            uncached_keys = []

            for j,item in enumerate(batch):
                if batch_keys:
                    cached_embedding = self._embedding_cache.get(batch_keys[j])
                    if cached_embedding is not None:
                        embeddings.append(cached_embedding)
                        continue
                    uncached_keys.append(batch_keys[j])
                batch_to_process.append(item)
                batch_indices.append(j + i)

            if not batch_to_process:
                continue

            batch_texts = [self._prepare_text_for_embedding(data) if isinstance(data, dict)
                           else str(data) for data in batch_to_process]

            # tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # generate embeddings
            outputs = self.model(**inputs)
            batch_embeddings = self._average_pooling(outputs.last_hidden_state, inputs['attention_mask'])

            if self.normalize_embeddings:
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

            batch_embeddings_np = batch_embeddings.detach().cpu().numpy()
            for idx, embedding in enumerate(batch_embeddings_np):
                if uncached_keys:
                    self._embedding_cache[uncached_keys[idx]] = embedding
                embeddings.append(embedding)

        return embeddings

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray):
        '''Compute cosine similarity between embeddings'''
        return float(F.cosine_similarity(
            torch.tensor(embedding1).unsqueeze(0),
            torch.tensor(embedding2).unsqueeze(0)
        ))

    def find_best_matches(self,
                          query_embedding: np.ndarray,
                          candidate_embeddings,
                          top_k: int = 5):
        """Find best matching candidates based on embedding similarity"""
        similarities = [
            self.compute_similarity(query_embedding, candidate)
            for candidate in candidate_embeddings
        ]

        # Get top-k matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(idx, similarities[idx]) for idx in top_indices]

    def clear_cache(self):
        """Clear the embedding cache"""
        self._embedding_cache.clear()


# example
if __name__ == "__main__":
    # Initialize embedder
    embedder = SupplyChainEmbedder()

    # Example makerspace data
    makerspace_data = {
        "title": "EcoEnergy Solutions",
        "description": "Provides solar panel installation and wind turbine services.",
        "keywords": ["renewable energy", "solar panels", "wind turbines"],
        "inventory-atoms": [
            {
                "identifier": "Solar Panels",
                "description": "Photovoltaic panels for harnessing solar energy."
            }
        ]
    }

    # Generate embedding
    embedding = embedder.generate_embedding(makerspace_data, cache_key="eco_energy")
    print(f"Generated embedding shape: {embedding.shape}")

    # Batch processing example
    batch_data = [makerspace_data] * 3
    batch_keys = ["eco1", "eco2", "eco3"]
    batch_embeddings = embedder.batch_generate_embeddings(batch_data, batch_keys)
    print(f"Generated {len(batch_embeddings)} batch embeddings")
