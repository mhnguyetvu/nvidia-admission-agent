# 📁 Document Storage

Place your PDF, Markdown, or text files here for ingestion into the chatbot.

## Recommended Structure

```
data/docs/
├── University1/
│   ├── Program1/
│   │   └── Fall_2026/
│   │       ├── admission_guide.pdf
│   │       └── requirements.pdf
│   └── Program2/
│       └── Spring_2027/
│           └── application_info.pdf
└── University2/
    └── ...
```

## Simple Structure (Also Works!)

```
data/docs/
├── admission_guide.pdf
├── requirements.pdf
└── scholarship_info.pdf
```

## Supported File Types

- ✅ **PDF** (.pdf) - Recommended for admission documents
- ✅ **Markdown** (.md) - Great for structured text
- ✅ **Text** (.txt) - Plain text files

## After Adding Documents

Run the ingestion script to process your files:

```bash
python scripts/ingest_docs.py
```

This will:
1. Extract text from all files in this folder (recursive)
2. Chunk the text into manageable pieces (500 characters each)
3. Generate embeddings using VNPay BGE m3
4. Store in ChromaDB at `./chroma_data/`

Your documents are then instantly searchable through the chatbot! 🎉

## Notes

- Files are processed recursively (all subfolders included)
- Duplicate PDFs are handled by their full path
- You can re-run ingestion anytime to add new documents
- Original files are never modified
