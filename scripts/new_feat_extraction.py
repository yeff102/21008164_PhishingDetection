import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from urllib.parse import urlparse, parse_qs
import re
from collections import Counter
import warnings
import time
from bs4 import BeautifulSoup
import ssl
import socket
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
warnings.filterwarnings('ignore')

class URLOnlyFeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        self.url_tfidf = TfidfVectorizer(max_features=30, analyzer='char', ngram_range=(2, 4))
        
        # Setup requests session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=2,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
    def extract_basic_url_features(self, url):
        """Extract basic features that can be derived from URL alone"""
        if pd.isna(url):
            return {}
        
        url_str = str(url)
        features = {}
        
        try:
            parsed_url = urlparse(url_str)
            
            # Basic URL features
            features['URLLength'] = len(url_str)
            features['DomainLength'] = len(parsed_url.netloc) if parsed_url.netloc else 0
            
            # TLD analysis
            domain_parts = parsed_url.netloc.split('.')
            tld = domain_parts[-1].lower() if domain_parts else ''
            known_tlds = ["com", "org", "net", "de", "edu", "gov", "mil", "int", "info", "biz", "co", "us", "uk", "ca", "au"]
            features['TLD_known'] = 1 if tld in known_tlds else 0
            
            # Character analysis
            features['CharContinuationRate'] = self.calculate_char_continuation_rate(url_str)
            features['LetterRatioInURL'] = sum(1 for c in url_str if c.isalpha()) / len(url_str) if url_str else 0
            features['SpacialCharRatioInURL'] = sum(1 for c in url_str if not c.isalnum() and c not in '.-/') / len(url_str) if url_str else 0
            
            # Subdomain count
            features['NoOfSubDomain'] = len(domain_parts) - 2 if len(domain_parts) > 2 else 0
            
            # HTTPS check
            features['IsHTTPS'] = 1 if parsed_url.scheme == 'https' else 0
            
            # Obfuscation detection
            features['HasObfuscation'] = 1 if self.detect_obfuscation(url_str) else 0
            features['ObfuscationRatio'] = self.calculate_obfuscation_ratio(url_str)
            
        except Exception as e:
            print(f"Error parsing URL {url_str}: {e}")
            # Set default values for basic features
            features.update({
                'URLLength': len(url_str),
                'DomainLength': 0,
                'TLD_known': 0,
                'CharContinuationRate': 0,
                'LetterRatioInURL': 0,
                'SpacialCharRatioInURL': 0,
                'NoOfSubDomain': 0,
                'IsHTTPS': 0,
                'HasObfuscation': 0,
                'ObfuscationRatio': 0
            })
        
        return features
    
    def fetch_website_content(self, url, timeout=10):
        """Safely fetch website content"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = self.session.get(url, headers=headers, timeout=timeout, verify=False, allow_redirects=True)
            response.raise_for_status()
            return response.text, response.status_code
            
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None, None
    
    def extract_content_features(self, url):
        """Extract features that require fetching website content"""
        features = {
            'URLSimilarityIndex': 0,
            'TLDLegitimateProb': 0.5,  # Default neutral probability
            'URLCharProb': 0.5,
            'HasTitle': 0,
            'DomainTitleMatchScore': 0,
            'URLTitleMatchScore': 0,
            'HasFavicon': 0,
            'Robots': 0,
            'IsResponsive': 0,
            'HasDescription': 0,
            'HasSocialNet': 0,
            'HasSubmitButton': 0,
            'HasHiddenFields': 0,
            'HasPasswordField': 0,
            'Bank': 0,
            'Pay': 0,
            'Crypto': 0,
            'HasCopyrightInfo': 0,
            'NoOfImage': 0,
            'NoOfCSS': 0,
            'NoOfJS': 0,
            'NoOfSelfRef': 0,
            'NoOfExternalRef': 0
        }
        
        try:
            content, status_code = self.fetch_website_content(url)
            
            if content:
                soup = BeautifulSoup(content, 'html.parser')
                parsed_url = urlparse(url)
                
                # Title analysis
                title_tag = soup.find('title')
                if title_tag and title_tag.text.strip():
                    features['HasTitle'] = 1
                    title_text = title_tag.text.strip().lower()
                    
                    # Domain-title match
                    domain = parsed_url.netloc.lower()
                    features['DomainTitleMatchScore'] = self.calculate_text_similarity(domain, title_text)
                    features['URLTitleMatchScore'] = self.calculate_text_similarity(url.lower(), title_text)
                
                # Meta description
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                features['HasDescription'] = 1 if meta_desc and meta_desc.get('content') else 0
                
                # Favicon check
                favicon_links = soup.find_all('link', rel=['icon', 'shortcut icon'])
                features['HasFavicon'] = 1 if favicon_links else 0
                
                # Form analysis
                forms = soup.find_all('form')
                for form in forms:
                    if form.find('input', {'type': 'submit'}) or form.find('button', {'type': 'submit'}):
                        features['HasSubmitButton'] = 1
                    if form.find('input', {'type': 'hidden'}):
                        features['HasHiddenFields'] = 1
                    if form.find('input', {'type': 'password'}):
                        features['HasPasswordField'] = 1
                
                # Content analysis
                text_content = soup.get_text().lower()
                
                # Banking/Payment keywords
                bank_keywords = ['bank', 'banking', 'account', 'login', 'signin']
                features['Bank'] = 1 if any(word in text_content for word in bank_keywords) else 0
                
                pay_keywords = ['payment', 'pay', 'paypal', 'credit card', 'debit']
                features['Pay'] = 1 if any(word in text_content for word in pay_keywords) else 0
                
                crypto_keywords = ['bitcoin', 'crypto', 'ethereum', 'blockchain']
                features['Crypto'] = 1 if any(word in text_content for word in crypto_keywords) else 0
                
                # Copyright info
                features['HasCopyrightInfo'] = 1 if 'copyright' in text_content or '©' in content else 0
                
                # Social networks
                social_keywords = ['facebook', 'twitter', 'instagram', 'linkedin', 'youtube']
                features['HasSocialNet'] = 1 if any(word in text_content for word in social_keywords) else 0
                
                # Resource counting
                features['NoOfImage'] = len(soup.find_all('img'))
                features['NoOfCSS'] = len(soup.find_all('link', rel='stylesheet'))
                features['NoOfJS'] = len(soup.find_all('script'))
                
                # Link analysis
                all_links = soup.find_all('a', href=True)
                internal_links = 0
                external_links = 0
                
                for link in all_links:
                    href = link['href']
                    if href.startswith('http'):
                        link_domain = urlparse(href).netloc
                        if link_domain == parsed_url.netloc:
                            internal_links += 1
                        else:
                            external_links += 1
                    elif href.startswith('/') or not href.startswith('http'):
                        internal_links += 1
                
                features['NoOfSelfRef'] = internal_links
                features['NoOfExternalRef'] = external_links
                
                # Robots.txt check
                try:
                    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
                    robots_response = self.session.get(robots_url, timeout=5)
                    features['Robots'] = 1 if robots_response.status_code == 200 else 0
                except:
                    features['Robots'] = 0
                
                # Responsive design (basic check)
                viewport_meta = soup.find('meta', attrs={'name': 'viewport'})
                features['IsResponsive'] = 1 if viewport_meta else 0
                
        except Exception as e:
            print(f"Error extracting content features from {url}: {e}")
        
        return features
    
    def calculate_char_continuation_rate(self, url):
        """Calculate character continuation rate"""
        if not url or len(url) < 2:
            return 0
        
        continuations = 0
        for i in range(len(url) - 1):
            if url[i] == url[i + 1]:
                continuations += 1
        
        return continuations / len(url) if len(url) > 0 else 0
    
    def detect_obfuscation(self, url):
        """Detect URL obfuscation patterns"""
        # IP address instead of domain
        if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
            return True
        
        # Excessive subdomains
        domain_part = urlparse(url).netloc
        if domain_part.count('.') > 3:
            return True
        
        # Suspicious characters
        if re.search(r'[%]|[0-9a-fA-F]{2}', url):
            return True
        
        return False
    
    def calculate_obfuscation_ratio(self, url):
        """Calculate obfuscation ratio"""
        suspicious_chars = len(re.findall(r'[%@#$&*()+=\[\]{}|\\:";\'<>?,./]', url))
        return suspicious_chars / len(url) if len(url) > 0 else 0
    
    def calculate_text_similarity(self, text1, text2):
        """Simple text similarity calculation"""
        if not text1 or not text2:
            return 0
        
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        
        if not set1 or not set2:
            return 0
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union) if union else 0
    
    def calculate_entropy(self, text):
        """Calculate Shannon entropy of text"""
        if not text:
            return 0
        
        counts = Counter(text)
        total = len(text)
        entropy = 0
        
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def extract_enhanced_features(self, url):
        """Extract enhanced features from the original script"""
        features = {}
        
        if pd.isna(url):
            return {f: 0 for f in ['url_has_ip', 'url_has_shortener', 'url_suspicious_tld', 
                                 'url_dash_count', 'url_underscore_count', 'url_subdomain_count',
                                 'url_path_depth', 'url_query_length', 'domain_entropy',
                                 'domain_vowel_ratio', 'domain_digit_ratio', 'domain_has_brand_words',
                                 'domain_consecutive_consonants', 'domain_typo_indicators',
                                 'title_has_suspicious_words', 'title_has_numbers', 'title_word_count',
                                 'title_char_count', 'title_exclamation_count', 'title_question_count',
                                 'has_robots_txt', 'has_description', 'is_https', 'has_favicon',
                                 'is_responsive', 'url_has_secure_words', 'url_has_banking_words',
                                 'form_complexity', 'external_resource_ratio', 'suspicious_score']}
        
        url_str = str(url)
        parsed_url = urlparse(url_str)
        url_clean = re.sub(r'https?://', '', url_str.lower())
        
        # URL n-gram features
        features['url_has_ip'] = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url_clean) else 0
        features['url_has_shortener'] = 1 if any(short in url_clean for short in ['bit.ly', 'tinyurl', 'goo.gl', 't.co']) else 0
        features['url_suspicious_tld'] = 1 if any(tld in url_clean for tld in ['.tk', '.ml', '.ga', '.cf']) else 0
        features['url_dash_count'] = url_clean.count('-')
        features['url_underscore_count'] = url_clean.count('_')
        features['url_subdomain_count'] = url_clean.count('.') - 1
        features['url_path_depth'] = url_clean.count('/')
        features['url_query_length'] = len(url_clean.split('?')[1]) if '?' in url_clean else 0
        
        # Domain features
        domain = parsed_url.netloc
        if domain:
            features['domain_entropy'] = self.calculate_entropy(domain)
            features['domain_vowel_ratio'] = sum(1 for c in domain.lower() if c in 'aeiou') / len(domain) if domain else 0
            features['domain_digit_ratio'] = sum(1 for c in domain if c.isdigit()) / len(domain) if domain else 0
            features['domain_has_brand_words'] = 1 if any(brand in domain.lower() for brand in ['paypal', 'amazon', 'google', 'microsoft', 'apple', 'facebook']) else 0
            
            consonant_matches = re.findall(r'[bcdfghjklmnpqrstvwxyz]+', domain.lower())
            features['domain_consecutive_consonants'] = max([len(match) for match in consonant_matches]) if consonant_matches else 0
            features['domain_typo_indicators'] = 1 if any(pattern in domain.lower() for pattern in ['payp4l', 'g00gle', 'amaz0n', 'micr0soft']) else 0
        else:
            features.update({
                'domain_entropy': 0,
                'domain_vowel_ratio': 0,
                'domain_digit_ratio': 0,
                'domain_has_brand_words': 0,
                'domain_consecutive_consonants': 0,
                'domain_typo_indicators': 0
            })
        
        # These will be filled by content analysis
        features.update({
            'title_has_suspicious_words': 0,
            'title_has_numbers': 0,
            'title_word_count': 0,
            'title_char_count': 0,
            'title_exclamation_count': 0,
            'title_question_count': 0,
            'has_robots_txt': 0,
            'has_description': 0,
            'is_https': 1 if parsed_url.scheme == 'https' else 0,
            'has_favicon': 0,
            'is_responsive': 0,
            'url_has_secure_words': 1 if any(word in url_str.lower() for word in ['secure', 'login', 'account', 'verify']) else 0,
            'url_has_banking_words': 1 if any(word in url_str.lower() for word in ['bank', 'payment', 'paypal', 'visa', 'mastercard']) else 0,
            'form_complexity': 0,
            'external_resource_ratio': 0,
            'suspicious_score': 0
        })
        
        return features
    
    def process_url_only_dataset(self, df, fetch_content=True, delay_between_requests=1):
        """Process dataset with only URL and label columns"""
        print(f"Processing {len(df)} URLs...")
        
        all_features = []
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"Processing {idx}/{len(df)} URLs...")
            
            url = row.get('URL', '') or row.get('url', '')  # Handle both cases
            
            # Extract basic features from URL
            features = self.extract_basic_url_features(url)
            
            # Extract enhanced features
            enhanced_features = self.extract_enhanced_features(url)
            features.update(enhanced_features)
            
            # Extract content features if requested
            if fetch_content and pd.notna(url):
                try:
                    content_features = self.extract_content_features(url)
                    features.update(content_features)
                    
                    # Update enhanced features with content information
                    if 'HasTitle' in content_features and content_features['HasTitle'] == 1:
                        # Try to get title for analysis (this would require storing it)
                        pass  # Title analysis would need to be done in extract_content_features
                    
                    features['has_robots_txt'] = content_features.get('Robots', 0)
                    features['has_description'] = content_features.get('HasDescription', 0)
                    features['has_favicon'] = content_features.get('HasFavicon', 0)
                    features['is_responsive'] = content_features.get('IsResponsive', 0)
                    
                    # Calculate behavioral features
                    features['form_complexity'] = (content_features.get('HasSubmitButton', 0) + 
                                                 content_features.get('HasHiddenFields', 0) + 
                                                 content_features.get('HasPasswordField', 0))
                    
                    features['external_resource_ratio'] = (content_features.get('NoOfExternalRef', 0) / 
                                                         max(1, content_features.get('NoOfExternalRef', 0) + 
                                                             content_features.get('NoOfSelfRef', 0)))
                    
                    features['suspicious_score'] = content_features.get('NoOfExternalRef', 0)  # Simplified
                    
                    time.sleep(delay_between_requests)  # Be respectful to servers
                    
                except Exception as e:
                    print(f"Error processing URL {url}: {e}")
            
            # Add label
            features['label'] = row.get('label', row.get('Label', 0))
            
            all_features.append(features)
        
        return pd.DataFrame(all_features)

def main():
    # Example usage
    extractor = URLOnlyFeatureExtractor()
    
    # Load your URL-only dataset
    # Replace with your actual file path
    url_dataset_path = "raw_data\openphish2021.csv"
    
    print("Loading URL-only dataset...")
    df_urls = pd.read_csv(url_dataset_path)
    
    # Process the URLs (set fetch_content=False for faster processing without web requests)
    processed_df = extractor.process_url_only_dataset(
        df_urls, 
        fetch_content=False,  # Set to False if you don't want to fetch webpage content
        delay_between_requests=0.5  # Delay between requests to be respectful
    )
    
    print(f"Processed dataset shape: {processed_df.shape}")
    print(f"Features extracted: {list(processed_df.columns)}")
    
    # Save processed dataset
    output_path = "processed_url_dataset.csv"
    processed_df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to: {output_path}")
    
    # Now you can combine this with your existing dataset
    # existing_dataset = pd.read_csv("path/to/enhanced_dataset.csv")
    # combined_dataset = pd.concat([existing_dataset, processed_df], ignore_index=True)
    # combined_dataset.to_csv("final_combined_dataset.csv", index=False)

if __name__ == "__main__":
    main()