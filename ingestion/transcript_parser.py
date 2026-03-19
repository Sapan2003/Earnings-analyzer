import re
from bs4 import BeautifulSoup
from utils.logger import get_logger

logger = get_logger("transcript_parser", "ingestion.log")


def clean_html(raw_text: str) -> str:
    """
    Removes HTML tags from raw SEC filing text.
    SEC filings come as HTML files - we need plain text.

    Args:
        raw_text: Raw HTML content from SEC filing

    Returns:
        Clean plain text
    """
    logger.debug("Cleaning HTML from filing text")

    try:
        soup = BeautifulSoup(raw_text, "html.parser")

        # Remove script and style tags completely
        for tag in soup(["script", "style", "head"]):
            tag.decompose()

        # Get plain text
        text = soup.get_text(separator=" ")

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        logger.debug(f"Cleaned text length: {len(text)} chars")
        return text

    except Exception as e:
        logger.error(f"Failed to clean HTML: {e}")
        return raw_text


def extract_sections(text: str) -> dict:
    """
    Extracts key sections from a 10-Q filing.
    We focus on the most valuable sections for financial analysis.

    Sections we extract:
    - MD&A: Management Discussion & Analysis (most valuable)
    - Risk Factors: What could go wrong
    - Financial Statements: Numbers

    Args:
        text: Clean plain text of filing

    Returns:
        Dictionary of section name -> section text
    """
    logger.debug("Extracting sections from filing")

    sections = {}

    # Key sections we want to find
    section_patterns = {
        "mdna": [
            r"management.{0,10}discussion.{0,10}analysis",
            r"item\s*2.*management"
        ],
        "risk_factors": [
            r"risk\s*factors",
            r"item\s*1a.*risk"
        ],
        "financial_statements": [
            r"financial\s*statements",
            r"item\s*1.*financial"
        ],
        "results_of_operations": [
            r"results\s*of\s*operations"
        ]
    }

    text_lower = text.lower()

    for section_name, patterns in section_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                # Extract text starting from where section begins
                start = match.start()
                # Take next 5000 characters as section content
                section_text = text[start:start + 5000]
                sections[section_name] = section_text
                logger.debug(f"Found section: {section_name}")
                break

    logger.debug(f"Extracted {len(sections)} sections")
    return sections


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Splits text into overlapping chunks for embedding.

    Why chunking?
    - LLMs have token limits - can't process 800k chars at once
    - Smaller chunks = more precise retrieval
    - Overlap ensures context isn't lost at chunk boundaries

    Args:
        text: Plain text to chunk
        chunk_size: Number of words per chunk
        overlap: Number of words to overlap between chunks

    Returns:
        List of text chunks
    """
    logger.debug(f"Chunking text | chunk_size={chunk_size} | overlap={overlap}")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        # Move forward by chunk_size minus overlap
        start += chunk_size - overlap

    logger.debug(f"Created {len(chunks)} chunks from {len(words)} words")
    return chunks


def parse_filing(filing: dict) -> list:
    """
    Main function - takes a raw filing and returns
    a list of chunks ready for embedding.

    Each chunk includes metadata so we know exactly
    which company, quarter, and section it came from.

    Args:
        filing: Dictionary with filing metadata + raw text

    Returns:
        List of chunk dictionaries with text + metadata
    """
    ticker = filing.get("ticker", "UNKNOWN")
    filed_date = filing.get("filed_date", "UNKNOWN")
    form_type = filing.get("form_type", "10-Q")

    logger.info(f"Parsing filing | ticker={ticker} | date={filed_date}")

    # Step 1 - Clean HTML
    clean_text = clean_html(filing["text"])

    # Step 2 - Extract key sections
    sections = extract_sections(clean_text)

    # If no sections found fall back to full text
    if not sections:
        logger.warning(f"No sections found for {ticker} {filed_date} - using full text")
        sections = {"full_text": clean_text}

    # Step 3 - Chunk each section
    all_chunks = []

    for section_name, section_text in sections.items():
        chunks = chunk_text(section_text)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "ticker": ticker,
                    "filed_date": filed_date,
                    "form_type": form_type,
                    "section": section_name,
                    "chunk_index": i
                }
            })

    logger.info(f"Parsed {len(all_chunks)} chunks for {ticker} {filed_date}")
    return all_chunks


if __name__ == "__main__":
    import sys
    from ingestion.sec_fetcher import fetch_company_filings

    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.transcript_parser <TICKER>")
        print("Example: python -m ingestion.transcript_parser AAPL")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    filings = fetch_company_filings(ticker, quarters=1)

    if filings:
        chunks = parse_filing(filings[0])
        print(f"Successfully parsed filing for {ticker}")
        print(f"Filing date: {filings[0]['filed_date']}")
        print(f"Total chunks created: {len(chunks)}")
        print(f"Sample text: {chunks[0]['text'][:200]}")
        print(f"Metadata: {chunks[0]['metadata']}")
    else:
        print(f"No filings found for {ticker}")