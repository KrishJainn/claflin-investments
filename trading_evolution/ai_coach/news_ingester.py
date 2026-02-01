"""
News Ingester for the AI Coach.

Fetches market news from multiple sources:
- NewsAPI (headlines and articles)
- Yahoo Finance (company news)
- Economic calendar events
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import logging
import json
import hashlib

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from ..ai_config import AIConfig, DEFAULT_AI_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """A single news item."""
    
    id: str  # Unique identifier (hash of content)
    headline: str
    source: str
    published_at: datetime
    url: str = ""
    summary: str = ""
    symbols: List[str] = field(default_factory=list)
    sentiment: float = 0.0  # -1 to 1, filled by SentimentAnalyzer
    impact: str = "medium"  # low, medium, high, critical
    category: str = "general"  # earnings, macro, sector, company
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'headline': self.headline,
            'source': self.source,
            'published_at': self.published_at.isoformat(),
            'url': self.url,
            'summary': self.summary,
            'symbols': self.symbols,
            'sentiment': self.sentiment,
            'impact': self.impact,
            'category': self.category,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'NewsItem':
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get('published_at'), str):
            data['published_at'] = datetime.fromisoformat(data['published_at'])
        return cls(**data)


class NewsIngester:
    """Fetches and manages market news from multiple sources."""
    
    # Mapping of Indian stock symbols to company names for news search
    SYMBOL_TO_COMPANY = {
        'RELIANCE.NS': 'Reliance Industries',
        'TCS.NS': 'Tata Consultancy Services TCS',
        'HDFCBANK.NS': 'HDFC Bank',
        'INFY.NS': 'Infosys',
        'ICICIBANK.NS': 'ICICI Bank',
        'HINDUNILVR.NS': 'Hindustan Unilever',
        'SBIN.NS': 'State Bank of India SBI',
        'BHARTIARTL.NS': 'Bharti Airtel',
        'ITC.NS': 'ITC Limited',
        'KOTAKBANK.NS': 'Kotak Mahindra Bank',
        'AXISBANK.NS': 'Axis Bank',
        'LT.NS': 'Larsen Toubro',
        'MARUTI.NS': 'Maruti Suzuki',
        'TATAMOTORS.NS': 'Tata Motors',
        'BAJFINANCE.NS': 'Bajaj Finance',
        'WIPRO.NS': 'Wipro',
        'TITAN.NS': 'Titan Company',
        'TATASTEEL.NS': 'Tata Steel',
    }
    
    # Keywords for market-wide news
    MARKET_KEYWORDS = [
        'Nifty 50', 'Sensex', 'NSE India', 'BSE India',
        'RBI interest rate', 'India GDP', 'India inflation',
        'FII investment India', 'Indian stock market',
    ]
    
    def __init__(self, config: AIConfig = None):
        """
        Initialize news ingester.
        
        Args:
            config: AI configuration
        """
        self.config = config or DEFAULT_AI_CONFIG
        self._seen_ids: Set[str] = set()
        self._news_cache: List[NewsItem] = []
        self._last_fetch: Optional[datetime] = None
        
        if not REQUESTS_AVAILABLE:
            logger.warning("requests library not available, NewsAPI disabled")
        
        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance library not available, Yahoo news disabled")
    
    def fetch_all(
        self,
        symbols: List[str] = None,
        hours_back: int = 24,
    ) -> List[NewsItem]:
        """
        Fetch news from all sources.
        
        Args:
            symbols: List of stock symbols to fetch news for
            hours_back: How many hours back to fetch
            
        Returns:
            List of deduplicated NewsItem objects
        """
        all_news = []
        
        # Fetch from NewsAPI if configured
        if self.config.news.newsapi_key and REQUESTS_AVAILABLE:
            newsapi_items = self._fetch_from_newsapi(symbols, hours_back)
            all_news.extend(newsapi_items)
        
        # Fetch from Yahoo Finance if enabled
        if self.config.news.yahoo_finance_enabled and YFINANCE_AVAILABLE:
            yahoo_items = self._fetch_from_yahoo(symbols)
            all_news.extend(yahoo_items)
        
        # Deduplicate
        unique_news = self._deduplicate(all_news)
        
        # Update cache
        self._news_cache = unique_news
        self._last_fetch = datetime.now()
        
        logger.info(f"Fetched {len(unique_news)} unique news items")
        return unique_news
    
    def _fetch_from_newsapi(
        self,
        symbols: List[str],
        hours_back: int,
    ) -> List[NewsItem]:
        """Fetch from NewsAPI."""
        news_items = []
        
        try:
            # Build search query
            queries = []
            
            # Add symbol-specific queries
            if symbols:
                for symbol in symbols[:5]:  # Limit to avoid rate limits
                    company = self.SYMBOL_TO_COMPANY.get(symbol, symbol.replace('.NS', ''))
                    queries.append(company)
            
            # Add market-wide query
            queries.append('Indian stock market OR Nifty OR Sensex')
            
            from_date = (datetime.now() - timedelta(hours=hours_back)).isoformat()
            
            for query in queries:
                url = 'https://newsapi.org/v2/everything'
                params = {
                    'q': query,
                    'from': from_date,
                    'sortBy': 'publishedAt',
                    'pageSize': min(self.config.news.max_articles_per_fetch // len(queries), 20),
                    'apiKey': self.config.news.newsapi_key,
                    'language': 'en',
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('articles', [])
                    
                    for article in articles:
                        news_id = self._generate_id(article.get('title', ''))
                        
                        item = NewsItem(
                            id=news_id,
                            headline=article.get('title', 'No title'),
                            source=article.get('source', {}).get('name', 'NewsAPI'),
                            published_at=self._parse_date(article.get('publishedAt')),
                            url=article.get('url', ''),
                            summary=article.get('description', '')[:500] if article.get('description') else '',
                            symbols=self._extract_symbols(article.get('title', '') + ' ' + (article.get('description') or '')),
                            category=self._categorize_news(article.get('title', '')),
                        )
                        news_items.append(item)
                else:
                    logger.warning(f"NewsAPI request failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"NewsAPI fetch error: {e}")
        
        return news_items
    
    def _fetch_from_yahoo(self, symbols: List[str]) -> List[NewsItem]:
        """Fetch from Yahoo Finance."""
        news_items = []
        
        if not symbols:
            return news_items
        
        try:
            for symbol in symbols[:10]:  # Limit to avoid slowdown
                try:
                    ticker = yf.Ticker(symbol)
                    news = ticker.news
                    
                    for item in (news or [])[:5]:  # Limit per symbol
                        news_id = self._generate_id(item.get('title', ''))
                        
                        published_ts = item.get('providerPublishTime', datetime.now().timestamp())
                        published_at = datetime.fromtimestamp(published_ts)
                        
                        news_item = NewsItem(
                            id=news_id,
                            headline=item.get('title', 'No title'),
                            source=item.get('publisher', 'Yahoo Finance'),
                            published_at=published_at,
                            url=item.get('link', ''),
                            summary=item.get('summary', '')[:500] if item.get('summary') else '',
                            symbols=[symbol],
                            category=self._categorize_news(item.get('title', '')),
                        )
                        news_items.append(news_item)
                        
                except Exception as e:
                    logger.debug(f"Failed to fetch Yahoo news for {symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"Yahoo Finance fetch error: {e}")
        
        return news_items
    
    def _generate_id(self, text: str) -> str:
        """Generate unique ID from text."""
        return hashlib.md5(text.encode()).hexdigest()[:12]
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime."""
        if not date_str:
            return datetime.now()
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            return datetime.now()
    
    def _deduplicate(self, news_items: List[NewsItem]) -> List[NewsItem]:
        """Remove duplicate news items."""
        unique = []
        seen_ids = set()
        
        for item in news_items:
            if item.id not in seen_ids and item.id not in self._seen_ids:
                unique.append(item)
                seen_ids.add(item.id)
                self._seen_ids.add(item.id)
        
        return unique
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract relevant stock symbols from text."""
        text_lower = text.lower()
        symbols = []
        
        for symbol, company in self.SYMBOL_TO_COMPANY.items():
            company_parts = company.lower().split()
            if any(part in text_lower for part in company_parts if len(part) > 3):
                symbols.append(symbol)
        
        return symbols
    
    def _categorize_news(self, headline: str) -> str:
        """Categorize news based on headline."""
        headline_lower = headline.lower()
        
        if any(word in headline_lower for word in ['earnings', 'profit', 'revenue', 'quarter', 'results']):
            return 'earnings'
        elif any(word in headline_lower for word in ['rbi', 'interest rate', 'gdp', 'inflation', 'policy']):
            return 'macro'
        elif any(word in headline_lower for word in ['sector', 'industry', 'market']):
            return 'sector'
        else:
            return 'company'
    
    def get_news_for_symbol(self, symbol: str) -> List[NewsItem]:
        """Get cached news for a specific symbol."""
        return [item for item in self._news_cache if symbol in item.symbols]
    
    def get_market_news(self) -> List[NewsItem]:
        """Get market-wide news (not symbol-specific)."""
        return [item for item in self._news_cache if item.category in ['macro', 'sector']]
    
    def get_high_impact_news(self) -> List[NewsItem]:
        """Get high and critical impact news."""
        return [item for item in self._news_cache if item.impact in ['high', 'critical']]
    
    def clear_cache(self):
        """Clear news cache."""
        self._news_cache = []
        self._seen_ids = set()
