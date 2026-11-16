
import re
from pathlib import Path
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent


def clean_text(text):
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Join words split across lines by PDF formatting
    text = re.sub(
        r"(?<=[a-z])-\s*\n\s*(?=[a-z])",
        "",
        text
    )

    # Preserve paragraph boundaries while joining ordinary line breaks
    paragraphs = re.split(r"\n\s*\n", text)
    cleaned_paragraphs = []

    for paragraph in paragraphs:
        paragraph = re.sub(r"\s*\n\s*", " ", paragraph)
        paragraph = re.sub(r"\s+", " ", paragraph).strip()

        if paragraph:
            cleaned_paragraphs.append(paragraph)

    return "\n\n".join(cleaned_paragraphs)


def extract_pdf(pdf_path):

    pdf_path = Path(pdf_path)

    if not pdf_path.is_absolute():
        pdf_path = BASE_DIR / pdf_path

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    reader = PdfReader(str(pdf_path), strict=False)

    if reader.is_encrypted:
        if not reader.decrypt(""):
            raise ValueError(
                f"Could not decrypt PDF: {pdf_path.name}"
            )

    extracted_pages = []
    references_started = False

    # Detect standalone bibliography headings before cleaning
    reference_pattern = re.compile(
        r"(?im)^[ \t]*(?:\d+[ \t]+)?"
        r"(?:references|bibliography|works cited)[ \t]*$"
    )

    for page_number, page in enumerate(reader.pages, start=1):

        raw_text = page.extract_text() or ""

        if not raw_text.strip():
            continue

        # Skip pages after the bibliography begins
        if references_started:
            continue

        # Find heading in original PDF text
        reference_match = reference_pattern.search(raw_text)

        # Clean page text
        page_text = clean_text(raw_text)

        # If bibliography starts on this page,
        # retain the content before the heading
        if reference_match:

            heading_pattern = re.compile(
                r"(?i)(?:references|bibliography|works cited)"
            )

            heading = heading_pattern.search(page_text)

            if heading:
                page_text = page_text[:heading.start()].strip()

            references_started = True

        if page_text:
            extracted_pages.append({
                "text": page_text,
                "source": pdf_path.name,
                "page": page_number
            })

    if not extracted_pages:
        raise ValueError(
            f"No extractable text found in {pdf_path.name}"
        )

    return extracted_pages


if __name__ == "__main__":

    documents_folder = BASE_DIR / "data" / "documents"
    pdf_files = sorted(documents_folder.glob("*.pdf"))

    for pdf_path in pdf_files:

        pages = extract_pdf(pdf_path)

        print(f"\nPDF: {pdf_path.name}")
        print(f"Extracted pages: {len(pages)}")

        for page in pages[:2]:
            print(f"\nPAGE {page['page']}")
            print(page["text"][:500])
