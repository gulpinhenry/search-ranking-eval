# search-ranking-eval

tested on torch 2.4.1+cu124
`pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124`
`bash install.sh`

cp hw3 content

Baselines:
BM25 - hw3
HNSW Bi-encoder - hw3
Cross-encoder 

BM25 + Bi-encoder reranking(cos) - hw3
BM25 + Cross-encoder reranking
embedding + HNSW + Cross-encoder reranking