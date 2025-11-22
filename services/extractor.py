import pdfplumber
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
from typing import List
import io

class DocumentExtractor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract(self, filename: str, file_content: bytes) -> str:
        filename = filename.lower()

        if filename.endswith(".pdf"):
            return self.extract_from_pdf(file_content)

        if filename.endswith(".docx"):
            return self.extract_from_docx(file_content)

        if filename.endswith(".csv"):
            return self.extract_from_csv(file_content)

        raise Exception("Unsupported file format")


    def extract_from_pdf(self, file_content: bytes) -> str:
        try:
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                if text.strip():
                    return text.strip()

            pdf_reader = PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            return text.strip() if text.strip() else "No text extracted"

        except Exception as e:
            raise Exception(f"Error extracting PDF: {str(e)}")

    def extract_from_docx(self, file_content: bytes) -> str:
        try:
            doc = Document(io.BytesIO(file_content))
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        except Exception as e:
            raise Exception(f"Error extracting DOCX: {str(e)}")

    def extract_from_excel(self, file_content: bytes) -> str:
        try:
            df = pd.read_excel(io.BytesIO(file_content))
            return df.to_string(index=False)
        except Exception as e:
            raise Exception(f"Error extracting Excel: {str(e)}")
