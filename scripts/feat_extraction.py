import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from urllib.parse import urlparse
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class PhishingFeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        self.url_tfidf = TfidfVectorizer(max_features=30, analyzer='char', ngram_range=(2, 4))
        
    def extract_url_ngrams(self, url):
        """Extract n-gram features from URL"""
        if pd.isna(url):
            return {}
        
        # Character n-grams
        url_clean = re.sub(r'https?://', '', str(url).lower())
        
        # Extract suspicious patterns
        features = {}
        features['url_has_ip'] = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url_clean) else 0
        features['url_has_shortener'] = 1 if any(short in url_clean for short in ['bit.ly', 'tinyurl', 'goo.gl', 't.co']) else 0
        features['url_suspicious_tld'] = 1 if any(tld in url_clean for tld in ['.tk', '.ml', '.ga', '.cf']) else 0
        features['url_dash_count'] = url_clean.count('-')
        features['url_underscore_count'] = url_clean.count('_')
        features['url_subdomain_count'] = url_clean.count('.') - 1
        features['url_path_depth'] = url_clean.count('/')
        features['url_query_length'] = len(url_clean.split('?')[1]) if '?' in url_clean else 0
        
        return features
    
    def extract_domain_features(self, url, domain):
        """Extract domain-based features"""
        if pd.isna(url) or pd.isna(domain):
            return {}
        
        features = {}
        parsed_url = urlparse(str(url))
        
        # Domain analysis
        features['domain_entropy'] = self.calculate_entropy(str(domain))
        features['domain_vowel_ratio'] = sum(1 for c in str(domain).lower() if c in 'aeiou') / len(str(domain)) if len(str(domain)) > 0 else 0
        features['domain_digit_ratio'] = sum(1 for c in str(domain) if c.isdigit()) / len(str(domain)) if len(str(domain)) > 0 else 0
        features['domain_has_brand_words'] = 1 if any(brand in str(domain).lower() for brand in ['paypal', 'amazon', 'google', 'microsoft', 'apple', 'facebook']) else 0
        
        # Suspicious domain patterns
        consonant_matches = re.findall(r'[bcdfghjklmnpqrstvwxyz]+', str(domain).lower())
        features['domain_consecutive_consonants'] = max([len(match) for match in consonant_matches]) if consonant_matches else 0
        features['domain_typo_indicators'] = 1 if any(pattern in str(domain).lower() for pattern in ['payp4l', 'g00gle', 'amaz0n', 'micr0soft']) else 0
        
        return features
    
    def extract_content_features(self, title, robots, has_description):
        """Extract content-based features"""
        features = {}
        
        # Title analysis
        if pd.notna(title):
            title_str = str(title).lower()
            features['title_has_suspicious_words'] = 1 if any(word in title_str for word in ['login', 'verify', 'update', 'suspend', 'urgent', 'click']) else 0
            features['title_has_numbers'] = 1 if any(c.isdigit() for c in title_str) else 0
            features['title_word_count'] = len(title_str.split())
            features['title_char_count'] = len(title_str)
            features['title_exclamation_count'] = title_str.count('!')
            features['title_question_count'] = title_str.count('?')
        else:
            features.update({
                'title_has_suspicious_words': 0,
                'title_has_numbers': 0,
                'title_word_count': 0,
                'title_char_count': 0,
                'title_exclamation_count': 0,
                'title_question_count': 0
            })
        
        # Other content features
        features['has_robots_txt'] = 1 if pd.notna(robots) and robots == 1 else 0
        features['has_description'] = 1 if pd.notna(has_description) and has_description == 1 else 0
        
        return features
    
    def extract_security_features(self, url, has_https, has_favicon, is_responsive):
        """Extract security and trust indicators"""
        features = {}
        
        features['is_https'] = 1 if has_https == 1 else 0
        features['has_favicon'] = 1 if has_favicon == 1 else 0
        features['is_responsive'] = 1 if is_responsive == 1 else 0
        
        # URL security patterns
        if pd.notna(url):
            url_str = str(url).lower()
            features['url_has_secure_words'] = 1 if any(word in url_str for word in ['secure', 'login', 'account', 'verify']) else 0
            features['url_has_banking_words'] = 1 if any(word in url_str for word in ['bank', 'payment', 'paypal', 'visa', 'mastercard']) else 0
        else:
            features['url_has_secure_words'] = 0
            features['url_has_banking_words'] = 0
        
        return features
    
    def extract_behavioral_features(self, row):
        """Extract behavioral pattern features"""
        features = {}
        
        # Form-related features
        features['form_complexity'] = (row.get('HasSubmitButton', 0) + 
                                     row.get('HasHiddenFields', 0) + 
                                     row.get('HasPasswordField', 0))
        
        # External references
        features['external_resource_ratio'] = (row.get('NoOfExternalRef', 0) / 
                                             max(1, row.get('NoOfExternalRef', 0) + 
                                                 row.get('NoOfSelfRef', 0)))
        
        # Suspicious indicators
        features['suspicious_score'] = (row.get('HasPopup', 0) + 
                                      row.get('NoOfiFrame', 0) + 
                                      row.get('HasExternalFormSubmit', 0))
        
        return features
    
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
    
    def extract_all_features(self, df):
        """Extract all features from the dataset"""
        print("Starting feature extraction...")
        
        new_features = []
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Processing row {idx}/{len(df)}")
            
            features = {}
            
            # Extract different types of features
            features.update(self.extract_url_ngrams(row.get('URL', '')))
            features.update(self.extract_domain_features(row.get('URL', ''), row.get('Domain', '')))
            features.update(self.extract_content_features(row.get('Title', ''), row.get('Robots', 0), row.get('HasDescription', 0)))
            features.update(self.extract_security_features(row.get('URL', ''), row.get('IsHTTPS', 0), row.get('HasFavicon', 0), row.get('IsResponsive', 0)))
            features.update(self.extract_behavioral_features(row))
            
            new_features.append(features)
        
        return pd.DataFrame(new_features)

def main():
    # Load dataset
    dataset_path = r"C:\Users\jasud\OneDrive - Sunway Education Group\school\sem 8\CP2\Phishing detection\outputs\datasets\preprocessed_dataset.csv"
    
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
    
    # Replace TLD with a binary 'TLD_known' feature
    known_tlds = ["com", "org", "net", "de", "edu", "gov", "mil", "int", "info", "biz", "co", "us", "uk", "ca", "au"]

    if 'TLD' in df.columns:
        print("\nProcessing TLD column to binary 'TLD_known'...")
        df['TLD_known'] = df['TLD'].apply(lambda x: 1 if str(x).lower() in known_tlds else 0)
        df.drop(columns=['TLD'], inplace=True)
        print(f"'TLD_known' column created. Distribution:\n{df['TLD_known'].value_counts()}")

    # Initialize feature extractor
    extractor = PhishingFeatureExtractor()
    
    # Extract new features
    new_features_df = extractor.extract_all_features(df)
    
    # Combine with existing features
    print("Combining features...")
    
    # Select important existing features to keep
    important_existing_features = [
        'URLLength', 'DomainLength', 'TLD_known', 'URLSimilarityIndex', 
        'CharContinuationRate', 'TLDLegitimateProb', 'URLCharProb',
        'NoOfSubDomain', 'HasObfuscation', 'ObfuscationRatio',
        'LetterRatioInURL', 'SpacialCharRatioInURL', 'IsHTTPS',
        'HasTitle', 'DomainTitleMatchScore', 'URLTitleMatchScore',
        'HasFavicon', 'Robots', 'IsResponsive', 'HasDescription',
        'HasSocialNet', 'HasSubmitButton', 'HasHiddenFields', 
        'HasPasswordField', 'Bank', 'Pay', 'Crypto', 'HasCopyrightInfo',
        'NoOfImage', 'NoOfCSS', 'NoOfJS', 'NoOfSelfRef', 'NoOfExternalRef',
        'label'
    ]
    
    # Keep only existing features that are in the dataset
    existing_features = [col for col in important_existing_features if col in df.columns]
    
    # Combine existing and new features
    combined_df = pd.concat([df[existing_features], new_features_df], axis=1)
    
    # Handle missing values
    combined_df = combined_df.fillna(0)
    
    print(f"Feature extraction complete!")
    print(f"Original features: {len(df.columns)}")
    print(f"New features: {len(new_features_df.columns)}")
    print(f"Total features: {len(combined_df.columns)}")
    
    # Save the enhanced dataset
    output_path = r"C:\Users\jasud\OneDrive - Sunway Education Group\school\sem 8\CP2\Phishing detection\outputs\datasets\enhanced_dataset.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"Enhanced dataset saved to: {output_path}")
    
    # Display feature summary
    print("\nFeature Summary:")
    print(f"- URL-based features: {len([f for f in new_features_df.columns if 'url_' in f])}")
    print(f"- Domain-based features: {len([f for f in new_features_df.columns if 'domain_' in f])}")
    print(f"- Content-based features: {len([f for f in new_features_df.columns if 'title_' in f or 'has_' in f])}")
    print(f"- Security features: {len([f for f in new_features_df.columns if any(s in f for s in ['https', 'favicon', 'responsive'])])}")
    print(f"- Behavioral features: {len([f for f in new_features_df.columns if any(s in f for s in ['form_', 'external_', 'suspicious_'])])}")

if __name__ == "__main__":
    main()