import json
import os
import re
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from datetime import datetime, timedelta
from urllib.parse import urljoin, quote
from collections import OrderedDict
import torch
import logging

# Setup logging
logging.basicConfig(
    filename='function_calls.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_function_call(func):
    """Decorator to log function calls and arguments."""
    def wrapper(*args, **kwargs):
        logging.info(f"Called {func.__qualname__} with args={args[1:]}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        logging.info(f"{func.__qualname__} returned {type(result).__name__}")
        return result
    return wrapper

class OTMCachedAgent:
    """
    Enhanced OTM API Agent with intelligent caching for frequently accessed data
    and provides intelligent API recommendations with actionable URLs.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", base_url: str = "https://otmgtm-test-gaeaotm.otmgtm.us-phoenix-1.ocs.oraclecloud.com"):
        self.model_name = model_name
        self.base_url = base_url
        
        # Cache configuration
        self.cache_config = {
            'max_size': 1000,  # Maximum cache entries
            'ttl_hours': 24,   # Time to live in hours
            'query_cache_size': 500,  # Query result cache
            'embedding_cache_size': 200  # Embedding cache
        }
        
        # Initialize caches
        self.query_cache = OrderedDict()
        self.embedding_cache = OrderedDict()
        self.url_cache = OrderedDict()
        self.stats_cache = OrderedDict()
        
        # Cache files
        self.cache_files = {
            'query_cache': 'otm_query_cache.pkl',
            'embedding_cache': 'otm_embedding_cache.pkl',
            'url_cache': 'otm_url_cache.pkl',
            'stats_cache': 'otm_stats_cache.pkl'
        }
        
        # OTM JSON files to process (use only the consolidated otm.json)
        self.otm_files = [
            "otm.json"
        ]
        
        # Vector database files
        self.vector_db_files = {
            'index': 'otm_vector_index.faiss',
            'metadata': 'otm_metadata.pkl',
            'embeddings': 'otm_embeddings.pkl'
        }
        
        # Initialize embedding model with proper error handling
        self.embedding_model = None
        self._initialize_embedding_model()
        
        # Initialize other components
        self.vector_db = None
        self.api_documents = []
        self.endpoint_metadata = {}
        self.faiss_index = None
        self.document_embeddings = None
        
        # Load existing caches
        self._load_caches()
        
        # Auto-load vector database on initialization
        self.load_vector_database()
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model with proper error handling"""
        try:
            # Force CPU usage to avoid meta tensor issues
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            self.embedding_model = SentenceTransformer(self.model_name, device='cpu')
        except Exception as e:
            try:
                # Try a simpler model
                self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cpu')
            except Exception as e2:
                raise Exception("Could not initialize any embedding model")
    
    def _load_caches(self):
        """Load existing caches from disk"""
        for cache_name, filename in self.cache_files.items():
            if os.path.exists(filename):
                try:
                    with open(filename, 'rb') as f:
                        cache_data = pickle.load(f)
                        setattr(self, cache_name, cache_data)
                except Exception as e:
                    pass
    
    def _save_caches(self):
        """Save caches to disk"""
        for cache_name, filename in self.cache_files.items():
            try:
                cache_data = getattr(self, cache_name)
                with open(filename, 'wb') as f:
                    pickle.dump(cache_data, f)
            except Exception as e:
                print(f"Error saving {cache_name}: {e}")
    
    def _get_cache_key(self, data: Any) -> str:
        """Generate cache key from data"""
        if isinstance(data, str):
            return hashlib.md5(data.encode()).hexdigest()
        elif isinstance(data, dict):
            return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
        else:
            return hashlib.md5(str(data).encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        if 'timestamp' not in cache_entry:
            return False
        
        cache_time = cache_entry['timestamp']
        current_time = datetime.now()
        age = current_time - cache_time
        
        return age.total_seconds() < (self.cache_config['ttl_hours'] * 3600)
    
    def _add_to_cache(self, cache: OrderedDict, key: str, value: Any, max_size: int = None):
        """Add entry to cache with LRU eviction"""
        if max_size is None:
            max_size = self.cache_config['max_size']
        
        # Remove old entry if exists
        if key in cache:
            del cache[key]
        
        # Add new entry
        cache[key] = {
            'value': value,
            'timestamp': datetime.now()
        }
        
        # Evict oldest if cache is full
        if len(cache) > max_size:
            cache.popitem(last=False)
    
    def _get_from_cache(self, cache: OrderedDict, key: str) -> Optional[Any]:
        """Get value from cache if valid"""
        if key not in cache:
            return None
        
        entry = cache[key]
        if not self._is_cache_valid(entry):
            del cache[key]
            return None
        
        # Move to end (LRU)
        cache.move_to_end(key)
        return entry['value']
    
    def get_cached_query_results(self, query: str) -> Optional[Dict]:
        """Get cached query results"""
        cache_key = self._get_cache_key(query)
        return self._get_from_cache(self.query_cache, cache_key)
    
    def cache_query_results(self, query: str, results: Dict):
        """Cache query results"""
        cache_key = self._get_cache_key(query)
        self._add_to_cache(self.query_cache, cache_key, results, self.cache_config['query_cache_size'])
    
    def get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding"""
        cache_key = self._get_cache_key(text)
        return self._get_from_cache(self.embedding_cache, cache_key)
    
    def cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache embedding"""
        cache_key = self._get_cache_key(text)
        self._add_to_cache(self.embedding_cache, cache_key, embedding, self.cache_config['embedding_cache_size'])
    
    def get_cached_url(self, endpoint_info: Dict, params: Dict = None) -> Optional[str]:
        """Get cached URL"""
        cache_key = self._get_cache_key({'endpoint': endpoint_info, 'params': params})
        return self._get_from_cache(self.url_cache, cache_key)
    
    def cache_url(self, endpoint_info: Dict, params: Dict, url: str):
        """Cache generated URL"""
        cache_key = self._get_cache_key({'endpoint': endpoint_info, 'params': params})
        self._add_to_cache(self.url_cache, cache_key, url)
    
    def get_cached_stats(self) -> Optional[Dict]:
        """Get cached statistics"""
        cache_key = "api_statistics"
        return self._get_from_cache(self.stats_cache, cache_key)
    
    def cache_stats(self, stats: Dict):
        """Cache API statistics"""
        cache_key = "api_statistics"
        self._add_to_cache(self.stats_cache, cache_key, stats)
    
    def load_otm_json_files(self) -> Dict[str, Any]:
        """Load and parse OTM JSON specification files"""
        all_specs = {}
        
        for filename in self.otm_files:
            if os.path.exists(filename):
                try:
                    print(f"Loading {filename}...")
                    with open(filename, 'r', encoding='utf-8') as f:
                        spec = json.load(f)
                        all_specs[filename] = spec
                        print(f"Loaded {filename} successfully")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            else:
                print(f"File {filename} not found")
        
        return all_specs
    
    def extract_api_documents(self, specs: Dict[str, Any]) -> List[Dict]:
        """Extract API endpoints and create searchable documents"""
        documents = []
        
        for filename, spec in specs.items():
            if 'paths' not in spec:
                continue
                
            print(f"Processing {filename} with {len(spec['paths'])} paths...")
            
            for path, methods in spec['paths'].items():
                for method, details in methods.items():
                    if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                        # Create comprehensive document for each endpoint
                        doc = self._create_endpoint_document(path, method, details, filename)
                        documents.append(doc)
                        
                        # Store metadata for quick access
                        endpoint_key = f"{method.upper()} {path}"
                        self.endpoint_metadata[endpoint_key] = {
                            'path': path,
                            'method': method.upper(),
                            'summary': details.get('summary', ''),
                            'description': details.get('description', ''),
                            'parameters': details.get('parameters', []),
                            'responses': details.get('responses', {}),
                            'tags': details.get('tags', []),
                            'source_file': filename
                        }
        
        print(f"Created {len(documents)} API documents")
        return documents
    
    def _create_endpoint_document(self, path: str, method: str, details: Dict, source_file: str) -> Dict:
        """Create a comprehensive document for an API endpoint"""
        
        # Extract all relevant information
        summary = details.get('summary', '')
        description = details.get('description', '')
        parameters = details.get('parameters', [])
        responses = details.get('responses', {})
        tags = details.get('tags', [])
        
        # Create parameter descriptions
        param_descriptions = []
        for param in parameters:
            param_desc = f"{param.get('name', '')}: {param.get('description', '')} ({param.get('type', 'unknown')})"
            param_descriptions.append(param_desc)
        
        # Create response descriptions
        response_descriptions = []
        for status_code, response in responses.items():
            response_desc = f"Status {status_code}: {response.get('description', '')}"
            response_descriptions.append(response_desc)
        
        # Create comprehensive text for embedding
        text_parts = [
            f"API Endpoint: {method.upper()} {path}",
            f"Summary: {summary}",
            f"Description: {description}",
            f"Tags: {', '.join(tags)}",
            f"Parameters: {'; '.join(param_descriptions)}",
            f"Responses: {'; '.join(response_descriptions)}"
        ]
        
        # Create document
        document = {
            'id': f"{method.upper()}_{path.replace('/', '_').replace('{', '').replace('}', '')}",
            'text': ' '.join(text_parts),
            'path': path,
            'method': method.upper(),
            'summary': summary,
            'description': description,
            'tags': tags,
            'parameters': parameters,
            'responses': responses,
            'source_file': source_file,
            'search_keywords': self._extract_keywords(path, summary, description, tags)
        }
        
        return document
    
    def _extract_keywords(self, path: str, summary: str, description: str, tags: List[str]) -> List[str]:
        """Extract relevant keywords for better search"""
        keywords = set()
        
        # Extract from path
        path_parts = path.lower().split('/')
        keywords.update(path_parts)
        
        # Extract from summary and description
        text = f"{summary} {description}".lower()
        # Common OTM keywords
        otm_keywords = [
            'order', 'shipment', 'trip', 'vehicle', 'driver', 'location', 'party',
            'rate', 'equipment', 'document', 'customs', 'hazmat', 'transport',
            'carrier', 'customer', 'vendor', 'pickup', 'delivery', 'route',
            'manifest', 'invoice', 'tracking', 'status', 'create', 'update',
            'delete', 'get', 'post', 'put', 'patch'
        ]
        
        for keyword in otm_keywords:
            if keyword in text:
                keywords.add(keyword)
        
        # Add tags
        keywords.update([tag.lower() for tag in tags])
        
        return list(keywords)
    
    @log_function_call
    def generate_actionable_url(self, endpoint_info: Dict, params: Dict = None) -> str:
        """Generate actionable URL for an API endpoint with caching"""
        print(f"[LOG] Function call: generate_actionable_url(endpoint_info={endpoint_info}, params={params})")
        
        # Check cache first
        cached_url = self.get_cached_url(endpoint_info, params)
        if cached_url:
            return cached_url
        
        path = endpoint_info['path']
        method = endpoint_info['method']
        
        # Parse the path to determine the correct OTM URL format
        if 'order' in path.lower():
            # Order-related URLs
            if params and 'orderId' in params:
                order_id = params['orderId']
                # Use the working OTM URL format for orders
                url = f"https://otmgtm-test-gaeaotm.otmgtm.us-phoenix-1.ocs.oraclecloud.com/GC3/glog.webserver.finder.WindowOpenFramesetServlet/nss?url=%2FGC3%2FOrderReleaseCustManagement%3Fmanager_layout_gid%3DORDER_RELEASE_WO_STOP%26pk%3DGAEA.{order_id}%26finder_set_gid%3DORDER_RELEASE_WO_STOP%26management_action%3Dedit"
            else:
                # General order management URL
                url = f"https://otmgtm-test-gaeaotm.otmgtm.us-phoenix-1.ocs.oraclecloud.com/GC3/OrderReleaseCustManagement"
        elif 'shipment' in path.lower():
            # Shipment-related URLs
            if params and 'orderId' in params:
                order_id = params['orderId']
                # Use the working OTM URL format for shipments
                url = f"https://otmgtm-test-gaeaotm.otmgtm.us-phoenix-1.ocs.oraclecloud.com/GC3/glog.webserver.finder.WindowOpenFramesetServlet/nss?url=%2FGC3%2Fglog.webserver.util.QueryResponseServlet%3Faction_name%3Dorder_release_history%26finder_set_gid%3DORDER_RELEASE_WO_STOP%26pk%3DGAEA.{order_id}"
            else:
                # General shipment management URL
                url = f"https://otmgtm-test-gaeaotm.otmgtm.us-phoenix-1.ocs.oraclecloud.com/GC3/ShipmentManagement"
        elif 'vehicle' in path.lower():
            # Vehicle-related URLs
            url = f"https://otmgtm-test-gaeaotm.otmgtm.us-phoenix-1.ocs.oraclecloud.com/GC3/VehicleManagement"
        elif 'driver' in path.lower():
            # Driver-related URLs
            url = f"https://otmgtm-test-gaeaotm.otmgtm.us-phoenix-1.ocs.oraclecloud.com/GC3/DriverManagement"
        elif 'rate' in path.lower() or 'pricing' in path.lower():
            # Rate/pricing-related URLs
            url = f"https://otmgtm-test-gaeaotm.otmgtm.us-phoenix-1.ocs.oraclecloud.com/GC3/RateManagement"
        elif 'document' in path.lower():
            # Document-related URLs
            url = f"https://otmgtm-test-gaeaotm.otmgtm.us-phoenix-1.ocs.oraclecloud.com/GC3/DocumentManagement"
        else:
            url = f"https://otmgtm-test-gaeaotm.otmgtm.us-phoenix-1.ocs.oraclecloud.com/GC3"
        
        # Cache the generated URL
        self.cache_url(endpoint_info, params, url)
        
        return url
    
    def extract_url_parameters(self, query: str) -> Dict[str, str]:
        """Extract potential URL parameters from user query"""
        params = {}
        
        # Extract order IDs
        order_patterns = [
            r'SX\d+',  # SX1750689449
            r'GAEA\.[A-Z0-9_]+',  # GAEA.SX1750689449
            r'\d{10,}',  # Any 10+ digit number
            r'[A-Z]{4}\.\d+',  # GAEA.57052
            r'[A-Z]+\.[A-Z0-9_]+',  # GAEA.STEPTENDER1_RR
        ]
        
        for pattern in order_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                params['orderId'] = matches[0]
                break
        
        # Extract other common parameters
        if 'limit' in query.lower():
            limit_match = re.search(r'limit[:\s]*(\d+)', query, re.IGNORECASE)
            if limit_match:
                params['limit'] = limit_match.group(1)
        
        if 'offset' in query.lower():
            offset_match = re.search(r'offset[:\s]*(\d+)', query, re.IGNORECASE)
            if offset_match:
                params['offset'] = offset_match.group(1)
        
        if 'status' in query.lower():
            status_match = re.search(r'status[:\s]*(\w+)', query, re.IGNORECASE)
            if status_match:
                params['status'] = status_match.group(1)
        
        return params
    
    def create_vector_database(self):
        """Create vector database from OTM API documents"""
        print("Creating OTM Vector Database...")
        
        # Check if database already exists and is valid
        if all(os.path.exists(f) for f in self.vector_db_files.values()):
            try:
                # Try to load existing database first
                test_index = faiss.read_index(self.vector_db_files['index'])
                with open(self.vector_db_files['metadata'], 'rb') as f:
                    test_metadata = pickle.load(f)
                print("Existing vector database is valid, using it...")
                return self.load_vector_database()
            except Exception as e:
                print(f"Existing database corrupted, recreating: {e}")
        
        # Load OTM JSON files
        specs = self.load_otm_json_files()
        if not specs:
            print("No OTM JSON files found!")
            return False
        
        # Extract API documents
        self.api_documents = self.extract_api_documents(specs)
        if not self.api_documents:
            print("No API documents extracted!")
            return False
        
        # Create embeddings with caching
        print("Creating embeddings...")
        texts = [doc['text'] for doc in self.api_documents]
        
        # Check cache for embeddings
        embeddings = []
        for text in texts:
            cached_embedding = self.get_cached_embedding(text)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                embedding = self.embedding_model.encode([text])[0]
                self.cache_embedding(text, embedding)
                embeddings.append(embedding)
        
        self.document_embeddings = np.array(embeddings)
        
        # Create FAISS index
        print("Creating FAISS index...")
        dimension = self.document_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.faiss_index.add(self.document_embeddings.astype('float32'))
        
        # Save vector database
        self._save_vector_database()
        
        print(f"Vector database created with {len(self.api_documents)} endpoints")
        return True
    
    def _save_vector_database(self):
        """Save vector database to files"""
        try:
            # Save FAISS index
            faiss.write_index(self.faiss_index, self.vector_db_files['index'])
            
            # Save metadata
            with open(self.vector_db_files['metadata'], 'wb') as f:
                pickle.dump({
                    'documents': self.api_documents,
                    'endpoint_metadata': self.endpoint_metadata,
                    'created_at': datetime.now().isoformat()
                }, f)
            
            # Save embeddings
            with open(self.vector_db_files['embeddings'], 'wb') as f:
                pickle.dump(self.document_embeddings, f)
            
            print("Vector database saved successfully")
        except Exception as e:
            print(f"Error saving vector database: {e}")
    
    def load_vector_database(self) -> bool:
        """Load existing vector database"""
        try:
            # Check if files exist
            if not all(os.path.exists(f) for f in self.vector_db_files.values()):
                print("Vector database files not found, creating new one...")
                return self.create_vector_database()
            
            # Load FAISS index
            self.faiss_index = faiss.read_index(self.vector_db_files['index'])
            
            # Load metadata
            with open(self.vector_db_files['metadata'], 'rb') as f:
                metadata = pickle.load(f)
                self.api_documents = metadata['documents']
                self.endpoint_metadata = metadata['endpoint_metadata']
            
            # Load embeddings
            with open(self.vector_db_files['embeddings'], 'rb') as f:
                self.document_embeddings = pickle.load(f)
            
            return True
        except Exception as e:
            # Don't recreate if loading fails, just return False
            return False
    
    @log_function_call
    def query_apis(self, user_query: str, top_k: int = 5) -> List[Dict]:
        """Query the vector database for relevant APIs with caching"""
        if not self.faiss_index or not self.api_documents:
            return []
        
        # Check cache first
        cached_results = self.get_cached_query_results(user_query)
        if cached_results:
            return cached_results
        
        # Encode user query with caching
        cached_embedding = self.get_cached_embedding(user_query)
        if cached_embedding is not None:
            query_embedding = cached_embedding.reshape(1, -1)
        else:
            query_embedding = self.embedding_model.encode([user_query])
            self.cache_embedding(user_query, query_embedding[0])
        
        # Search in FAISS index
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        
        # Get results with deduplication by method+path
        results = []
        seen_keys = set()
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.api_documents):
                doc = self.api_documents[idx]
                uniq_key = (doc['method'], doc['path'])
                if uniq_key in seen_keys:
                    continue
                seen_keys.add(uniq_key)

                # Extract parameters from query
                params = self.extract_url_parameters(user_query)

                # Generate actionable URL
                actionable_url = self.generate_actionable_url({
                    'path': doc['path'],
                    'method': doc['method']
                }, params)

                result = {
                    'rank': i + 1,
                    'score': float(score),
                    'endpoint_info': {
                        'path': doc['path'],
                        'method': doc['method']
                    },
                    'summary': doc['summary'],
                    'description': doc['description'],
                    'tags': doc['tags'],
                    'parameters': doc['parameters'],
                    'responses': doc['responses'],
                    'source_file': doc['source_file'],
                    'keywords': doc['search_keywords'],
                    'actionable_url': actionable_url,
                    'extracted_params': params
                }
                results.append(result)
        
        # Cache the results
        self.cache_query_results(user_query, results)
        
        return results
    
    @log_function_call
    def get_api_suggestions(self, query: str) -> Dict[str, Any]:
        """Get comprehensive API suggestions for a user query with caching"""
        results = self.query_apis(query, top_k=10)
        
        if not results:
            return {
                'success': False,
                'message': 'No relevant APIs found for your query.',
                'suggestions': []
            }
        
        # Group by category
        categories = {
            'exact_matches': [],
            'high_relevance': [],
            'related': []
        }
        
        for result in results:
            if result['score'] > 0.8:
                categories['exact_matches'].append(result)
            elif result['score'] > 0.6:
                categories['high_relevance'].append(result)
            else:
                categories['related'].append(result)
        
        # Create response
        response = {
            'success': True,
            'query': query,
            'total_found': len(results),
            'categories': categories,
            'top_recommendations': results[:3],
            'suggested_queries': self._generate_suggested_queries(query, results)
        }
        
        return response
    
    def _generate_suggested_queries(self, original_query: str, results: List[Dict]) -> List[str]:
        """Generate suggested follow-up queries"""
        suggestions = []
        
        # Extract common themes from results
        all_tags = []
        all_keywords = []
        
        for result in results[:5]:  # Top 5 results
            all_tags.extend(result['tags'])
            all_keywords.extend(result['keywords'])
        
        # Find most common themes
        tag_counts = {}
        keyword_counts = {}
        
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Generate suggestions based on common themes
        common_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        common_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for tag, count in common_tags:
            if tag.lower() not in original_query.lower():
                suggestions.append(f"Show me {tag} related APIs")
        
        for keyword, count in common_keywords:
            if keyword.lower() not in original_query.lower():
                suggestions.append(f"Find APIs for {keyword}")
        
        return suggestions[:5]  # Return top 5 suggestions
    
    @log_function_call
    def get_api_statistics(self) -> Dict[str, Any]:
        """Get statistics about the API database with caching"""
        # Check cache first
        cached_stats = self.get_cached_stats()
        if cached_stats:
            return cached_stats
        
        if not self.api_documents:
            return {'error': 'No API documents loaded'}
        
        # Count by method
        method_counts = {}
        tag_counts = {}
        file_counts = {}
        
        for doc in self.api_documents:
            method = doc['method']
            method_counts[method] = method_counts.get(method, 0) + 1
            
            for tag in doc['tags']:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            source = doc['source_file']
            file_counts[source] = file_counts.get(source, 0) + 1
        
        stats = {
            'total_endpoints': len(self.api_documents),
            'methods': method_counts,
            'tags': dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'source_files': file_counts,
            'database_size_mb': self._get_database_size(),
            'cache_stats': self._get_cache_statistics()
        }
        
        # Cache the statistics
        self.cache_stats(stats)
        
        return stats
    
    def _get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'query_cache_size': len(self.query_cache),
            'embedding_cache_size': len(self.embedding_cache),
            'url_cache_size': len(self.url_cache),
            'stats_cache_size': len(self.stats_cache),
            'total_cache_entries': len(self.query_cache) + len(self.embedding_cache) + len(self.url_cache) + len(self.stats_cache)
        }
    
    def _get_database_size(self) -> float:
        """Get the size of the vector database in MB"""
        total_size = 0
        for filename in self.vector_db_files.values():
            if os.path.exists(filename):
                total_size += os.path.getsize(filename)
        return round(total_size / (1024 * 1024), 2)
    
    @log_function_call
    def search_by_tags(self, tags: List[str]) -> List[Dict]:
        """Search APIs by specific tags"""
        if not self.api_documents:
            return []
        
        results = []
        for doc in self.api_documents:
            if any(tag.lower() in [t.lower() for t in doc['tags']] for tag in tags):
                # Generate actionable URL
                actionable_url = self.generate_actionable_url({
                    'path': doc['path'],
                    'method': doc['method']
                })
                
                results.append({
                    'endpoint': f"{doc['method']} {doc['path']}",
                    'summary': doc['summary'],
                    'tags': doc['tags'],
                    'path': doc['path'],
                    'method': doc['method'],
                    'actionable_url': actionable_url
                })
        
        return results
    
    @log_function_call
    def get_endpoint_details(self, endpoint_key: str) -> Optional[Dict]:
        """Get detailed information about a specific endpoint"""
        if endpoint_key in self.endpoint_metadata:
            endpoint_info = self.endpoint_metadata[endpoint_key]
            # Generate actionable URL
            actionable_url = self.generate_actionable_url(endpoint_info)
            endpoint_info['actionable_url'] = actionable_url
            return endpoint_info
        return None
    
    def clear_cache(self, cache_type: str = None):
        """Clear cache entries"""
        if cache_type is None:
            # Clear all caches
            self.query_cache.clear()
            self.embedding_cache.clear()
            self.url_cache.clear()
            self.stats_cache.clear()
            print("All caches cleared")
        elif cache_type == 'query':
            self.query_cache.clear()
            print("Query cache cleared")
        elif cache_type == 'embedding':
            self.embedding_cache.clear()
            print("Embedding cache cleared")
        elif cache_type == 'url':
            self.url_cache.clear()
            print("URL cache cleared")
        elif cache_type == 'stats':
            self.stats_cache.clear()
            print("Stats cache cleared")
    
    def __del__(self):
        """Save caches when object is destroyed"""
        self._save_caches()

# FastAPI integration
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn

# FastAPI models
class QueryRequest(BaseModel):
    query: str

class EndpointResult(BaseModel):
    path: str
    method: str
    summary: str
    description: str
    actionable_url: str

# FastAPI app
api_app = FastAPI()

# CORS for OpenAI
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Limit this in prod!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Reuse the agent
agent_instance = OTMCachedAgent()

@api_app.post("/query_otm", response_model=List[EndpointResult])
def query_otm(request: QueryRequest):
    results = agent_instance.query_apis(request.query)
    return [
        EndpointResult(
            path=r["endpoint_info"]["path"],
            method=r["endpoint_info"]["method"],
            summary=r["summary"],
            description=r["description"],
            actionable_url=r["actionable_url"]
        ) for r in results
    ]