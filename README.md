# Methods-Compare
Sliding window,CRAG and RAG methods compare

In this project, data previously embedded using three different techniques — Sliding Window, CRAG , and standard RAG — is stored in separate vector databases.

The current system does not perform any re-embedding, but instead queries the already existing vector stores based on user questions. Each retrieval method queries its corresponding vector store, and the system generates responses accordingly.

Each method operates with its own retrieval logic:

Sliding Window: Documents were split into overlapping fixed-size chunks before embedding.

CRAG: Uses an LLM-based scoring mechanism to select the most relevant chunks based on question-context alignment.

RAG: Applies traditional similarity-based retrieval without additional filtering.

The names of the vector databases used for each technique are clearly defined within the code. This setup allows for direct comparison of retrieval quality and answer accuracy across the different methods.



