from typing import List, Dict, Any
from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter
import json
import glob

def create_document_content(data: Dict) -> str:
    content_parts = []
    content_parts.append(f"Title: {data.get('title', '')}")
    content_parts.append(f"Description: {data.get('description', '')}")
    content_parts.append(f"Keywords: {', '.join(data.get('keywords', []))}")
    
    for atom_type in ['inventory-atoms', 'product-atoms', 'tool-list-atoms']:
        atoms = data.get(atom_type, [])
        if atoms:
            atom_descriptions = [f"{atom.get('identifier', '')}: {atom.get('description', '')}" 
                               for atom in atoms]
            content_parts.append(f"{atom_type.title()}: {', '.join(atom_descriptions)}")
    
    return '\n'.join(content_parts)
    

def load_okw_json(path: str) -> List[Document]:
    documents = []
    for file_path in glob.glob(f'{path}/*'):
        with open(file_path, 'r') as f:
            data = json.load(f)
            content = create_document_content(data)
            doc = Document(page_content=content, metadata={'title':data.get('title', '')})
            documents.append(doc)
    return documents
    

class SemanticJSONSplitter(TextSplitter):
    def __init__(self, max_chunk_size: int=1000):
        super().__init__()
        self.max_chunk_size = max_chunk_size

    def split_text(self, text:str) -> List[str]:
        '''req. abstract method'''
        return [text]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        final_chunks = []
        for doc in documents:
            chunks = self._split_document(doc)
            final_chunks.extend(chunks)
        return final_chunks

    def _split_document(self, document: Document) -> List[Document]:
        content_parts = document.page_content.split('\n')
        json_structure={}

        for part in content_parts:
            if part.startswith('Title: '):
                json_structure['title'] = part[7:]
            elif part.startswith('Description: '):
                json_structure['description'] = part[13:]
            elif part.startswith('Keywords: '):
                json_structure['keywords'] = [k.strip() for k in part[10:].split(',')]

        chunks = []
        core_chunk = {
            'title': json_structure.get('title', ''),
            'description': json_structure.get('description', ''),
            'keywords': json_structure.get('keywords', [])
        }
        chunks.append(Document(
            page_content=self._format_chunk(core_chunk),
            metadata={
                'title': json_structure.get('title', ''),
                'chunk_type': 'core'
            }
        ))

        atom_types = ['inventory-atoms', 'product-atoms', 'tool_list-atoms']
        for atom_type in atom_types:
            atoms = self._extract_atoms(content_parts, atom_type)
            if atoms:
                atom_chunks = self._chunk_atoms(atoms, atom_type, json_structure.get('title', ''))
                chunks.extend(atom_chunks)
        return chunks

    def _extract_atoms(self, content_parts: List[str], atom_type:str) -> List[Dict[str, str]]:
        atoms = []
        for part in content_parts:
            if part.startswith(f'{atom_type.title()}: '):
                atom_texts = part[part.find(': ')+2:].split(', ')
                for atom_text in atom_texts:
                    if ': ' in atom_text:
                        identifier, description = atom_text.split(': ', 1)
                        atoms.append({
                            'identifier': identifier,
                            'description': description
                        })
        return atoms

    def _chunk_atoms(self, atoms: List[Dict[str, str]], atom_type: str, title: str) -> List[Document]:
        chunks = []
        current_chunk = []
        current_size = 0
        
        for atom in atoms:
            atom_text = f"{atom['identifier']}: {atom['description']}"
            atom_size = len(atom_text)
            
            if current_size + atom_size > self.max_chunk_size and current_chunk:
                chunks.append(self._create_atom_document(current_chunk, atom_type, title))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(atom)
            current_size += atom_size
        
        if current_chunk:
            chunks.append(self._create_atom_document(current_chunk, atom_type, title))
        
        return chunks
    
    def _create_atom_document(self, atoms: List[Dict[str, str]], atom_type: str, title: str) -> Document:
        content = self._format_chunk({
            'title': title,
            atom_type: atoms
        })
        return Document(
            page_content=content,
            metadata={
                'title': title,
                'chunk_type': atom_type
            }
        )
    
    def _format_chunk(self, chunk_data: Dict[str, Any]) -> str:
        content_parts = []
        
        if 'title' in chunk_data:
            content_parts.append(f"Title: {chunk_data['title']}")
        
        if 'description' in chunk_data:
            content_parts.append(f"Description: {chunk_data['description']}")
        
        if 'keywords' in chunk_data:
            content_parts.append(f"Keywords: {', '.join(chunk_data['keywords'])}")
        
        for atom_type in ['inventory-atoms', 'product-atoms', 'tool-list-atoms']:
            if atom_type in chunk_data:
                atoms_text = ', '.join([f"{atom['identifier']}: {atom['description']}"
                                      for atom in chunk_data[atom_type]])
                content_parts.append(f"{atom_type.title()}: {atoms_text}")
        
        return '\n'.join(content_parts)


def load_and_process_documents(directory_path: str) -> List[Document]:
    splitter = SemanticJSONSplitter(max_chunk_size=500)
    documents = load_okw_json(directory_path)
    return splitter.split_documents(documents)
            
