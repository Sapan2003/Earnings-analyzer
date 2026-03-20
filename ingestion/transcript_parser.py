import re
from bs4 import BeautifulSoup
from utils.logger import get_logger

logger = get_logger("transcript_parser", "ingestion.log")


def extract_financial_tables(soup) -> str:
    """
    Extracts financial tables from SEC filing HTML.
    Converts HTML tables to readable text format
    so numbers stay connected to their labels.

    This is critical for capturing specific revenue
    figures like "Microsoft Cloud revenue | $38.9B"
    that get lost when we strip HTML naively.

    Args:
        soup: BeautifulSoup object of filing HTML

    Returns:
        String of all financial tables in readable format
    """
    tables_text = []
    tables = soup.find_all("table")

    for table in tables:
        rows = table.find_all("tr")

        # Skip tiny tables (navigation, headers etc)
        if len(rows) < 3:
            continue

        table_lines = []
        for row in rows:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue

            cell_texts = []
            for cell in cells:
                text = cell.get_text(strip=True)
                # Clean up common formatting artifacts
                text = re.sub(r'\s+', ' ', text).strip()
                if text:
                    cell_texts.append(text)

            # Only keep rows with actual content
            if len(cell_texts) >= 2:
                table_lines.append(" | ".join(cell_texts))

        # Only keep tables with enough meaningful rows
        if len(table_lines) >= 3:
            tables_text.append("\n".join(table_lines))

    result = "\n\n---TABLE---\n\n".join(tables_text)
    logger.debug(f"Extracted {len(tables_text)} financial tables")
    return result


def clean_html(raw_text: str) -> str:
    """
    Cleans HTML and preserves financial table data.

    Two-pass approach:
    1. Extract financial tables BEFORE stripping HTML
       so numbers stay connected to their labels
    2. Extract narrative text separately
    3. Combine both for complete coverage

    Args:
        raw_text: Raw HTML content from SEC filing

    Returns:
        Clean text with financial tables preserved
    """
    logger.debug("Cleaning HTML from filing text")

    try:
        soup = BeautifulSoup(raw_text, "html.parser")

        # Pass 1 - Extract tables BEFORE stripping HTML
        tables_text = extract_financial_tables(soup)

        # Pass 2 - Remove script and style tags
        for tag in soup(["script", "style", "head"]):
            tag.decompose()

        # Get narrative plain text
        narrative_text = soup.get_text(separator=" ")
        narrative_text = re.sub(r'\s+', ' ', narrative_text)
        narrative_text = narrative_text.strip()

        # Combine narrative + tables
        if tables_text:
            combined = (narrative_text +
                       "\n\nFINANCIAL TABLES:\n" +
                       tables_text)
        else:
            combined = narrative_text

        logger.debug(f"Cleaned text length: {len(combined)} chars "
                    f"(narrative + {len(tables_text)} chars tables)")
        return combined

    except Exception as e:
        logger.error(f"Failed to clean HTML: {e}")
        return raw_text


def extract_sections(text: str) -> dict:
    """
    Extracts key sections from a 10-Q or 10-K filing.

    Args:
        text: Clean plain text of filing

    Returns:
        Dictionary of section name -> section text
    """
    logger.debug("Extracting sections from filing")
    sections = {}

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
    found_positions = []

    for section_name, patterns in section_patterns.items():
        for pattern in patterns:
            matches = list(re.finditer(pattern, text_lower))
            if matches:
                # Skip first match (table of contents)
                # Use second match which is actual content
                match = matches[1] if len(matches) > 1 else matches[0]
                start = match.start()
                found_positions.append((start, section_name))
                break

    # Sort sections by position in document
    found_positions.sort(key=lambda x: x[0])

    for i, (start, section_name) in enumerate(found_positions):
        # End at next section start or 25000 chars
        end = (found_positions[i + 1][0]
               if i + 1 < len(found_positions)
               else start + 25000)
        end = min(end, start + 25000)
        section_text = text[start:end]
        sections[section_name] = section_text
        logger.debug(f"Found section: {section_name} | "
                    f"length: {len(section_text)} chars")

    # Also extract financial tables section separately
    # Tables are appended at end of clean_html output
    if "FINANCIAL TABLES:" in text:
        tables_start = text.index("FINANCIAL TABLES:")
        tables_text = text[tables_start:]
        sections["financial_tables"] = tables_text
        logger.debug(f"Added financial tables section | "
                    f"length: {len(tables_text)} chars")

    logger.debug(f"Extracted {len(sections)} sections")
    return sections


def chunk_text(text: str,
               chunk_size: int = 500,
               overlap: int = 50) -> list:
    """
    Splits text into overlapping chunks for embedding.

    Args:
        text: Plain text to chunk
        chunk_size: Number of words per chunk
        overlap: Number of words to overlap between chunks

    Returns:
        List of text chunks
    """
    logger.debug(f"Chunking text | chunk_size={chunk_size} | "
                f"overlap={overlap}")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    logger.debug(f"Created {len(chunks)} chunks from "
                f"{len(words)} words")
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

    logger.info(f"Parsing filing | ticker={ticker} | "
               f"date={filed_date} | type={form_type}")

    # Step 1 - Clean HTML and extract tables
    clean_text = clean_html(filing["text"])

    # Step 2 - Extract key sections
    sections = extract_sections(clean_text)

    # Fallback to full text if no sections found
    if not sections:
        logger.warning(f"No sections found for {ticker} "
                      f"{filed_date} - using full text")
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

    logger.info(f"Parsed {len(all_chunks)} chunks for "
               f"{ticker} {filed_date}")
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
        print(f"Form type: {filings[0]['form_type']}")
        print(f"Total chunks created: {len(chunks)}")
        print(f"Sections found: "
              f"{set(c['metadata']['section'] for c in chunks)}")
        print(f"Sample text: {chunks[0]['text'][:200]}")
    else:
        print(f"No filings found for {ticker}")