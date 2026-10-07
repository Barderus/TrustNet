from docx import Document
import pdfplumber


def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])


def extract_text_from_pdf(file, max_pages=20):
    text = ""
    with pdfplumber.open(file) as pdf:
        if len(pdf.pages) > max_pages:
            raise ValueError(f"PDFs are limited to {max_pages} pages.")
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()
