# MCAL_Code_Generation

Loading and Splitting PDF Text
The function load_and_split_pdf_text(pdf_path) is used to load and split the text from a PDF document into manageable chunks.

Extracting Tables from PDFs
The function extract_tables_from_pdf(pdf_path) extracts tables from a PDF document and converts them to HTML format.

Embedding Text and Tables
The code uses SentenceTransformer to embed the text and table data for similarity search.

Retrieving Text and Tables
The MultiVectorRetriever class is used to retrieve relevant text and table data based on a query. It uses FAISS for efficient similarity search.

Generating C and H Code
The function generate_mcal_code(query) generates the C and H code for a given query using the Ollama LLM model.
