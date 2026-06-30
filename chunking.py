from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Takes a list of documents and chunks them appropriately.
    For markdown, it tries to preserve header context.
    For other text, it uses recursive character splitting.
    """
    
    final_chunks = []
    
    # Base splitter for large blocks of text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150
    )
    
    # Splitter specifically for maintaining Markdown hierarchies
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    for doc in documents:
        source = doc.metadata.get('source', '')
        
        if source.endswith('.md'):
            # For Markdown, split by headers first so chunks remain semantically related
            md_header_splits = markdown_splitter.split_text(doc.page_content)
            
            # Re-attach original metadata (like file path, page number) to the new chunks
            for split in md_header_splits:
                split.metadata.update(doc.metadata)
            
            # If the sections between headers are STILL too big, break them down safely
            splits = text_splitter.split_documents(md_header_splits)
            final_chunks.extend(splits)
            
        else:
            # For PDFs and plain TXT files
            splits = text_splitter.split_documents([doc])
            final_chunks.extend(splits)
            
    return final_chunks
