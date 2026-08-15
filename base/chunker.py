# using recursive chunking - split at natural boundaies or para/sentences 
# not semantic chunking -> increases accuracy / heavy load -> can test later

from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(pages_data, chunk_size = 500, overlap = 50):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = overlap,
        separators= ["\n\n", "\n", ". ", " ", ""] # priority high to low
    )

    chunks = []
    chunk_id = 0

    for page in pages_data:
        page_num = page["page"]
        split_texts = splitter.split_text(page["text"])

        for text in split_texts:
            chunks.append({
                "chunk_id":chunk_id,
                "page" : page_num,
                "text": text
            })
            chunk_id+=1
    return chunks


if __name__ == "__main__":
    from read_pdf import load_pdf

    path = "../data/climate_paper.pdf"
    pages = load_pdf(path)

    chunks = chunk_text(pages, chunk_size=650, overlap=50)

    print(chunks[0])
    print(chunks[1])
    print(chunks[2])
    print(chunks[3])
    print(chunks[4])