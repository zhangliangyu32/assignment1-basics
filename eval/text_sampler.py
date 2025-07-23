#!/usr/bin/env python3
"""
Simple script to sample 10 documents from the OWT dataset.
"""

import random

def main():
    # Read and collect documents (separated by <eof> tokens)
    documents = []
    current_doc = []
    
    with open("../data/TinyStoriesV2-GPT4-train.txt", 'r') as f:
        for line in f:
            if not line.strip():  # Document separator
                if current_doc:  # If we have content, save as document
                    documents.append(''.join(current_doc))
                    current_doc = []
                    if len(documents) == 10:  # Stop after 10 documents
                        break
            else:
                current_doc.append(line)
        
        # Don't forget the last document if file doesn't end with <eof>
        if current_doc and len(documents) < 10:
            documents.append(''.join(current_doc))
    
    print(f"Found {len(documents)} documents")
    
    # Save samples to file
    with open("sampled_tiny_stories.txt", 'w') as f:
        for i, doc in enumerate(documents, 1):
            f.write(doc)
            f.write("\n")

    print(f"Saved {len(documents)} documents to sampled_tiny_stories.txt")
    
    documents = []
    current_doc = []

    with open("../data/owt_valid.txt", 'r') as f:
        for line in f:
            if line.strip() == "-":  # Document separator
                if current_doc:  # If we have content, save as document
                    documents.append(''.join(current_doc))
                    current_doc = []
                    if len(documents) == 10:  # Stop after 10 documents
                        break
            else:
                current_doc.append(line)
        
        # Don't forget the last document if file doesn't end with <eof>
        if current_doc and len(documents) < 10:
            documents.append(''.join(current_doc))
    
    print(f"Found {len(documents)} documents")
    
    # Save samples to file
    with open("sampled_owt.txt", 'w') as f:
        for i, doc in enumerate(documents, 1):
            f.write(doc)
            f.write("\n")

    print(f"Saved {len(documents)} documents to sampled_owt.txt")
if __name__ == "__main__":
    main()
