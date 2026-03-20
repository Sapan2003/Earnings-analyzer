import requests
import json
import time
from utils.logger import get_logger

logger = get_logger("sec_fetcher", "ingestion.log")

HEADERS = {
    "User-Agent": "earnings-analyzer sapanparikh13@gmail.com"
}

def get_cik(ticker: str) -> str | None:
    """
    Converts a stock ticker (e.g. AAPL) to a CIK number.
    CIK is the unique ID SEC uses to identify every company.
    """
    logger.info(f"Looking up CIK for ticker: {ticker}")
    
    url = "https://www.sec.gov/files/company_tickers.json"
    
    try:
        response = requests.get(url, headers=HEADERS)
        data = response.json()
        
        for entry in data.values():
            if entry["ticker"].upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                logger.info(f"Found CIK for {ticker}: {cik}")
                return cik
        
        logger.warning(f"No CIK found for ticker: {ticker}")
        return None
    
    except Exception as e:
        logger.error(f"Failed to get CIK for {ticker}: {e}")
        return None


def get_filings(cik: str, form_type: str = "10-Q", count: int = 8) -> list:
    """
    Fetches the most recent filings for a company.
    
    Args:
        cik: Company CIK number
        form_type: Type of filing (10-Q or 10-K)
        count: Number of filings to fetch
    
    Returns:
        List of filing metadata dictionaries
    """
    logger.info(f"Fetching {count} {form_type} filings for CIK: {cik}")
    
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    
    try:
        response = requests.get(url, headers=HEADERS)
        data = response.json()
        
        filings = data["filings"]["recent"]
        
        results = []
        for i, form in enumerate(filings["form"]):
            if form == form_type:
                results.append({
                    "form_type": form,
                    "filed_date": filings["filingDate"][i],
                    "accession_number": filings["accessionNumber"][i],
                    "primary_document": filings["primaryDocument"][i],
                    "cik": cik
                })
            
            if len(results) == count:
                break
        
        logger.info(f"Found {len(results)} {form_type} filings")
        return results
    
    except Exception as e:
        logger.error(f"Failed to fetch filings for CIK {cik}: {e}")
        return []


def fetch_filing_text(cik: str, accession_number: str, primary_document: str) -> str | None:
    """
    Downloads the actual text content of a filing.
    
    Args:
        cik: Company CIK number
        accession_number: Filing accession number
        primary_document: Primary document filename
    
    Returns:
        Raw text content of the filing
    """
    accession_clean = accession_number.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_clean}/{primary_document}"
    
    logger.info(f"Downloading filing: {primary_document}")
    logger.debug(f"URL: {url}")
    
    try:
        response = requests.get(url, headers=HEADERS)
        
        if response.status_code == 200:
            logger.info(f"Successfully downloaded: {primary_document}")
            return response.text
        else:
            logger.warning(f"Failed to download {primary_document}: status {response.status_code}")
            return None
    
    except Exception as e:
        logger.error(f"Failed to download filing: {e}")
        return None


def fetch_company_filings(ticker: str, quarters: int = 8) -> list:
    """
    Fetches both 10-Q quarterly and 10-K annual filings.

    10-Q: quarterly performance, revenue, earnings
    10-K: annual report with geographic breakdown,
          complete risk factors, full financial statements

    Args:
        ticker: Stock ticker symbol (e.g. AAPL)
        quarters: Number of quarterly filings to fetch

    Returns:
        List of dictionaries with filing metadata + text content
    """
    logger.info(f"=== Starting filing fetch for {ticker} ===")

    # Step 1 - Get CIK
    cik = get_cik(ticker)
    if not cik:
        logger.error(f"Cannot proceed - no CIK found for {ticker}")
        return []

    # Step 2 - Get quarterly filing metadata
    quarterly = get_filings(cik, form_type="10-Q", count=quarters)
    logger.info(f"Found {len(quarterly)} 10-Q filings")

    # Step 3 - Get annual filing metadata (last 2 years)
    annual = get_filings(cik, form_type="10-K", count=2)
    logger.info(f"Found {len(annual)} 10-K filings")

    all_filings = quarterly + annual

    if not all_filings:
        logger.error(f"No filings found for {ticker}")
        return []

    # Step 4 - Download each filing text
    results = []
    for filing in all_filings:
        text = fetch_filing_text(
            cik,
            filing["accession_number"],
            filing["primary_document"]
        )

        if text:
            filing["text"] = text
            filing["ticker"] = ticker.upper()
            results.append(filing)

        # Be polite to SEC servers
        time.sleep(0.5)

    logger.info(f"=== Completed: fetched {len(results)} filings "
               f"({len(quarterly)} quarterly + {len(annual)} annual) "
               f"for {ticker} ===")
    return results


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    quarters = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    filings = fetch_company_filings(ticker, quarters=quarters)
    if filings:
        print(f"Successfully fetched {len(filings)} filings for {ticker}")
        for f in filings:
            print(f"  - {f['form_type']} filed on {f['filed_date']} | "
                  f"text length: {len(f['text'])} chars")
    else:
        print(f"No filings fetched for {ticker}")