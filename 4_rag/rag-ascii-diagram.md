# RAG System Flow Diagram

## Phase 1: Document Indexing & Embedding

```
                                    ┌─────────────┐
                                    │     PDF     │  10M Tokens
                                    └─────────────┘
                                           │
                     ┌─────────────────────┼─────────────────────┐
                     │                     │                     │
                     ▼                     ▼                     ▼
              ┌───────────┐         ┌───────────┐         ┌───────────┐
              │   Chunk   │         │   Chunk   │   ...   │   Chunk   │
              └───────────┘         └───────────┘         └───────────┘
                     │                     │                     │
                     │              Each chunk = 1K Tokens       │
                     │                     │                     │
                     ▼                     ▼                     ▼
                   ┌───┐                 ┌───┐                 ┌───┐
                   │ []│                 │ []│                 │ []│
                   └───┘                 └───┘                 └───┘
                     │                     │                     │
              LLM Embedder Convert Text to Embedding (This part cost $)
                     │                     │                     │
                     ▼                     ▼                     ▼
              ┌───────────┐         ┌───────────┐         ┌───────────┐
              │  [1,0,    │         │  [1,0,    │   ...   │  [1,0,    │
              │   1,0]    │         │   1,0]    │         │   1,0]    │
              └───────────┘         └───────────┘         └───────────┘
                     │                     │                     │
                     └─────────────────────┼─────────────────────┘
                                           ▼
                                  ┌─────────────────┐
                                  │  Vector Store   │
                                  │                 │
                                  │ Store Embeddings│
                                  └─────────────────┘
```

**Embedding = numerical representation of text**

### Example Chunk

```
┌──────────────────────────────────────┐
│  ***                                 │
│  "Harry - yer a wizard,"             │
│  Hagrid had a twinkle in             │
│  his eyes as he said it,             │
│  a hint of the joy he                │
│  felt at being the one               │
│  to reveal Harry's true              │
│  identity.                           │
│  ***                                 │
└──────────────────────────────────────┘
```

### Example Embedding

```
┌──────────────────────────────────────┐
│  Dog   = [1,1,2,0,1]                 │
│  Cat   = [1,1,2,3,2]                 │
│  House = [8,3,7,1,9]                 │
└──────────────────────────────────────┘
```

---

## Phase 2: RAG Q&A Flow

```
                                  ┌─────────────────┐
                                  │  Vector Store   │
                                  │                 │
                                  │ Store Embeddings│
                                  └────────┬────────┘
                                           │
                                          (2)
                                           │
                                           ▼
┌─────┐        ┌──────────┐       ┌───┐       ┌──────────┐         ┌──────────────┐
│ user│------->│ Question │------>│ []│------>│  [1,0,   │-------->│              │
└──┬──┘        └──────────┘       └───┘       │   1,0]   │         │  Retriever   │
   │                                          └──────────┘         │              │
   │           ┌─────────────────────┐                             │      x       │
   │           │      ChatGPT        │                             └──────┬───────┘
   │           │                     │                                    │
   ▲           │  ┌───────────────┐  │                                    │
   │           │  │               │  │                                    │
   │           │  │  Chunk        │  │                                    ▼
   │           │  │               │  │                              ┌─────────────┐
   │           │  │  ┌─────────┐  │  │                              │    Chunk    │
   │           │  │  │ Chunk   │  │  │                              │ Chunk 1 Text│
   │           │  │  │ 1 Text  │  │  │                              └─────────────┘
   │           │  │  └─────────┘  │  │                                     │
   │           │  │               │  │                                     │
   │           │  │  Chunk 2 Text │  │                                     │
   │           │  │               │  │                              ┌─────────────┐
   │           │  │  ┌─────────┐  │  │                              │    Chunk    │
   │           │  │  │ Chunk   │<-┼──┼──────────────────────────────│             │
   │           │  │  │ 3 Text  │  │  │                              │ Chunk 3 Text│
   │           │  │  └─────────┘  │  │                              └─────────────┘
   │           │  │               │  │
   │           │  │  Question     │  │
   │           │  │               │  │                              ┌─────────────┐
   │           │  │  ┌─────────┐  │  │                              │  Question   │
   └───────────┼──┼──│ Chunk   │  │  │                              │             │
               │  │  │         │  │  │                              │    Chunk    │
               │  │  └─────────┘  │  │                              └─────────────┘
               │  │               │  │                                     │
               │  │  ***          │  │                                     │
               │  └───────────────┘  │                                     │
               └─────────────────────┘                                     ▼
                          ▲                                          ┌─────────────┐
                          │                                          │     ***     │
                          └──────────────────────────────────────────┘─────────────┘
```

---

## Complete RAG Flow Summary

1. **Document Processing**: PDF split into chunks → Each chunk embedded → Stored in vector database
2. **Query Processing**: User question → Question embedded → Similar chunks retrieved
3. **Answer Generation**: Retrieved chunks + Question → Sent to ChatGPT → Generate answer
