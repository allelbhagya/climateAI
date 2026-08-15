# taking pdf and splitting text, page indexed

from pypdf import PdfReader

def load_pdf(path):

    reader = PdfReader(path)
    pages_data = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages_data.append({
                "page":i+1,
                "text":text
            })
    return pages_data



if __name__ == "__main__":
    path = "../data/climate_paper.pdf"
    pages = load_pdf(path)

    print(pages[0]["text"])