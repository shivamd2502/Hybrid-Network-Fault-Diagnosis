"""
NETWORK TROUBLESHOOTING INFERENCE MODULE
=========================================
NLP preprocessing + Decision Tree inference for diagnosis

Features:
---------
- Symptom text to features conversion
- User query processing
- Model prediction with confidence scores
- Multi-turn conversation state management

Author: AI Assistant
Date: 2026
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# NLP libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    print("[Warning] scikit-learn not available")


# ============================================================================
# STEP 1: SYMPTOM KEYWORD MAPPING
# ============================================================================

SYMPTOM_KEYWORDS = {
    "ping_gateway": {
        "yes_keywords": ["can ping gateway", "gateway responds", "gateway is reachable", "ping gateway works"],
        "no_keywords": ["cannot ping gateway", "gateway unreachable", "no ping", "gateway not responding"]
    },
    "has_ip": {
        "yes_keywords": ["has ip", "ip address assigned", "got ip", "configured ip"],
        "no_keywords": ["no ip", "no ip address", "missing ip", "169.254", "self-assigned", "apipa"]
    },
    "ping_ip": {
        "yes_keywords": ["can ping ip", "ip responds", "ping ip works", "ip reachable"],
        "no_keywords": ["cannot ping ip", "ip unreachable", "ping ip fails", "no ping"]
    },
    "ping_domain": {
        "yes_keywords": ["can access websites", "websites work", "google works", "internet accessible"],
        "no_keywords": ["cannot access websites", "websites fail", "google fails", "internet not working"]
    },
    "ip_conflict": {
        "yes_keywords": ["duplicate ip", "ip conflict", "ip already in use", "same ip"],
        "no_keywords": ["no duplicate", "no conflict", "unique ip"]
    }
}

NETWORK_TYPE_KEYWORDS = {
    "WiFi": ["wifi", "wireless", "wlan", "802.11"],
    "Ethernet": ["ethernet", "wired", "lan", "cable"]
}

OS_TYPE_KEYWORDS = {
    "Windows": ["windows", "win10", "win11", "cmd", "ipconfig"],
    "macOS": ["mac", "osx", "macos", "apple", "ifconfig"],
    "Linux": ["linux", "ubuntu", "terminal", "bash"]
}


# ============================================================================
# STEP 2: FEATURE EXTRACTION FROM TEXT
# ============================================================================

class SymptomExtractor:
    """
    Extract diagnostic features from user symptom description
    """
    
    def __init__(self):
        self.symptoms = SYMPTOM_KEYWORDS
        self.network_types = NETWORK_TYPE_KEYWORDS
        self.os_types = OS_TYPE_KEYWORDS
    
    def extract_binary_feature(self, text: str, feature_name: str) -> Optional[int]:
        """
        Extract binary feature (0 or 1) from text
        
        Parameters:
        -----------
        text : str
            User input text
        feature_name : str
            Feature name (ping_gateway, has_ip, etc.)
        
        Returns:
        --------
        Optional[int] : 0 (no), 1 (yes), or None (unknown)
        """
        
        text_lower = text.lower()
        
        if feature_name in self.symptoms:
            yes_keywords = self.symptoms[feature_name]["yes_keywords"]
            no_keywords = self.symptoms[feature_name]["no_keywords"]
            
            # Check for positive indicators
            for keyword in yes_keywords:
                if keyword in text_lower:
                    return 1
            
            # Check for negative indicators
            for keyword in no_keywords:
                if keyword in text_lower:
                    return 0
        
        return None
    
    def extract_network_type(self, text: str) -> str:
        """
        Extract network type from text (WiFi or Ethernet)
        
        Parameters:
        -----------
        text : str
            User input
        
        Returns:
        --------
        str : "WiFi", "Ethernet", or "Unknown"
        """
        
        text_lower = text.lower()
        
        for net_type, keywords in self.network_types.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return net_type
        
        return "Unknown"
    
    def extract_os_type(self, text: str) -> str:
        """
        Extract OS type from text
        
        Parameters:
        -----------
        text : str
            User input
        
        Returns:
        --------
        str : "Windows", "macOS", "Linux", or "Unknown"
        """
        
        text_lower = text.lower()
        
        for os_type, keywords in self.os_types.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return os_type
        
        return "Unknown"
    
    def extract_features(self, text: str) -> Dict:
        """
        Extract all features from symptom text
        
        Parameters:
        -----------
        text : str
            User symptom description
        
        Returns:
        --------
        Dict : Extracted features
        """
        
        features = {
            'symptom_text': text,
            'ping_gateway': self.extract_binary_feature(text, 'ping_gateway'),
            'has_ip': self.extract_binary_feature(text, 'has_ip'),
            'ping_ip': self.extract_binary_feature(text, 'ping_ip'),
            'ping_domain': self.extract_binary_feature(text, 'ping_domain'),
            'ip_conflict': self.extract_binary_feature(text, 'ip_conflict'),
            'network_type': self.extract_network_type(text),
            'os_type': self.extract_os_type(text),
            'recently_updated': 0,  # Can be asked as follow-up
            'vpn_enabled': 0,  # Can be asked as follow-up
            'firewall_enabled': 0,  # Can be asked as follow-up
        }
        
        return features


# ============================================================================
# STEP 3: CONVERSATION STATE MANAGEMENT
# ============================================================================

class ConversationState:
    """
    Manage multi-turn conversation state
    """
    
    def __init__(self):
        self.features = {}
        self.history = []
        self.extracted_text = None
    
    def add_message(self, role: str, content: str):
        """
        Add message to conversation history
        
        Parameters:
        -----------
        role : str
            "user" or "bot"
        content : str
            Message content
        """
        
        self.history.append({'role': role, 'content': content})
    
    def update_features(self, new_features: Dict):
        """
        Update feature values
        
        Parameters:
        -----------
        new_features : Dict
            Features to update
        """
        
        self.features.update(new_features)
    
    def get_missing_features(self) -> List[str]:
        """
        Get list of missing (None) feature values
        
        Returns:
        --------
        List[str] : List of feature names with None values
        """
        
        missing = [k for k, v in self.features.items() if v is None]
        return missing
    
    def reset(self):
        """Reset conversation state"""
        self.features = {}
        self.history = []
        self.extracted_text = None


# ============================================================================
# STEP 4: MODEL INFERENCE ENGINE
# ============================================================================

class NetworkTroubleshootingInference:
    """
    Complete inference pipeline:
    1. Load model and encoders
    2. Process user input (NLP)
    3. Generate diagnosis
    4. Return solutions
    """
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialize inference engine
        
        Parameters:
        -----------
        model_dir : str
            Directory containing saved model and encoders
        """
        
        self.model_dir = Path(model_dir)
        self.model = None
        self.encoders = None
        self.metadata = None
        self.symptom_extractor = SymptomExtractor()
        self.conversation_state = ConversationState()
        
        # Load model and components
        self._load_model()
    
    def _load_model(self):
        """Load trained model and encoders"""
        
        try:
            # Load model
            with open(self.model_dir / 'dt_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            # Load encoders
            with open(self.model_dir / 'encoders.pkl', 'rb') as f:
                self.encoders = pickle.load(f)
            
            # Load metadata
            with open(self.model_dir / 'metadata.json', 'r') as f:
                self.metadata = json.load(f)
            
            print(f"✓ Model loaded successfully from {self.model_dir}")
            
        except FileNotFoundError as e:
            print(f"✗ Error loading model: {e}")
            print(f"  Ensure model files exist in {self.model_dir}")
            raise
    
    def process_user_input(self, text: str) -> Dict:
        """
        Process user input and extract features
        
        Parameters:
        -----------
        text : str
            User symptom description
        
        Returns:
        --------
        Dict : Extracted features
        """
        
        # Extract features from text
        features = self.symptom_extractor.extract_features(text)
        
        # Update conversation state
        self.conversation_state.update_features(features)
        self.conversation_state.extracted_text = text
        
        return features
    
    def fill_missing_features(self, responses: Dict):
        """
        Fill missing features from follow-up questions
        
        Parameters:
        -----------
        responses : Dict
            Responses to follow-up questions
        """
        
        for feature, value in responses.items():
            if feature in self.conversation_state.features:
                self.conversation_state.features[feature] = value
    
    def _prepare_features_for_prediction(self) -> pd.DataFrame:
        """
        Prepare features for model prediction
        
        Returns:
        --------
        pd.DataFrame : Properly formatted and encoded features
        """
        
        features = self.conversation_state.features.copy()
        
        # Default missing values to 0
        feature_names = self.metadata['feature_names']
        for feature in feature_names:
            if feature not in features or features[feature] is None:
                features[feature] = 0
        
        # Create DataFrame with correct column order
        df = pd.DataFrame([features])[feature_names]
        
        # Encode categorical features
        for col in self.encoders:
            if col in df.columns:
                # Get the encoder
                encoder = self.encoders[col]
                
                # Handle unknown categories
                if df[col].iloc[0] not in encoder.classes_:
                    df[col].iloc[0] = encoder.classes_[0]  # Use default
                
                # Encode
                df[col] = encoder.transform(df[col])
        
        return df
    
    def predict(self) -> Tuple[str, float, List[str]]:
        """
        Generate diagnosis prediction
        
        Returns:
        --------
        Tuple : (diagnosis, confidence, solutions)
        """
        
        # Prepare features
        X = self._prepare_features_for_prediction()
        
        # Get prediction
        prediction = self.model.predict(X)[0]
        
        # Get prediction probability
        probabilities = self.model.predict_proba(X)[0]
        confidence = float(np.max(probabilities))
        
        # Get solutions
        solutions = self._get_solutions(prediction)
        
        return prediction, confidence, solutions
    
    def _get_solutions(self, diagnosis: str) -> List[str]:
        """
        Get recommended solutions for diagnosis
        
        Parameters:
        -----------
        diagnosis : str
            Diagnosed issue
        
        Returns:
        --------
        List[str] : List of solution steps
        """
        
        # Solutions mapping (same as in dataset generation)
        solutions_map = {
            "Router Issue": [
                "Check if router is powered on",
                "Restart router (unplug 30 seconds)",
                "Check Wi-Fi signal strength",
                "Check if other devices can connect",
                "Restart network adapter",
                "Update router firmware",
                "Reset router or contact ISP"
            ],
            "DNS Issue": [
                "Change DNS to 8.8.8.8 (Google DNS)",
                "Flush DNS cache (ipconfig /flushdns)",
                "Restart network adapter",
                "Test with nslookup google.com",
                "Try Cloudflare DNS (1.1.1.1)",
                "Check router DNS settings"
            ],
            "IP Conflict": [
                "Identify device with conflicting IP (arp -a)",
                "Release current IP (ipconfig /release)",
                "Renew DHCP lease (ipconfig /renew)",
                "Restart network adapter",
                "Configure static IP if needed",
                "Check DHCP pool size in router",
                "Restart router"
            ],
            "DHCP Failure": [
                "Check DHCP is enabled in router",
                "Release and renew IP (ipconfig /release && ipconfig /renew)",
                "Restart router",
                "Check device is in DHCP scope",
                "Configure static IP as workaround",
                "Check router DHCP logs"
            ],
            "Gateway Unreachable": [
                "Verify default gateway (route print)",
                "Ping default gateway",
                "Configure correct gateway",
                "Restart TCP/IP stack",
                "Check gateway device is powered on",
                "Restart network adapter",
                "Check router WAN IP configuration"
            ],
            "Network Adapter Issue": [
                "Check if adapter is disabled in Device Manager",
                "Enable network adapter",
                "Check ethernet cable connection",
                "Restart network adapter",
                "Update adapter drivers",
                "Uninstall and reinstall drivers",
                "Replace hardware if needed"
            ],
            "Subnet Mismatch": [
                "Check device IP (ipconfig)",
                "Check gateway IP and subnet mask",
                "Verify device is on same subnet",
                "Use subnet calculator",
                "Configure correct subnet mask",
                "Check DHCP subnet assignment",
                "Restart network to apply settings"
            ],
            "DNS Timeout": [
                "Test DNS resolution (nslookup)",
                "Change to faster DNS server",
                "Flush DNS cache",
                "Check if DNS server is reachable",
                "Increase DNS timeout",
                "Check ISP DNS status",
                "Restart router to reset DNS",
                "Check firewall DNS rules"
            ]
        }
        
        return solutions_map.get(diagnosis, ["No specific solutions available"])
    
    def get_follow_up_questions(self) -> List[str]:
        """
        Get list of follow-up questions for missing features
        
        Returns:
        --------
        List[str] : Follow-up questions
        """
        
        questions = {
            'ping_gateway': "Can you ping your default gateway? (yes/no)",
            'has_ip': "Does your device have an IP address assigned? (yes/no)",
            'ping_ip': "Can you ping an IP address directly? (yes/no)",
            'ping_domain': "Can you access websites like google.com? (yes/no)",
            'ip_conflict': "Are you seeing a duplicate IP conflict warning? (yes/no)",
            'network_type': "Are you using WiFi or Ethernet? (wifi/ethernet)",
            'os_type': "What operating system? (windows/mac/linux)",
            'recently_updated': "Have you recently updated your system? (yes/no)",
            'vpn_enabled': "Do you have VPN enabled? (yes/no)",
            'firewall_enabled': "Do you have a firewall enabled? (yes/no)"
        }
        
        missing = self.conversation_state.get_missing_features()
        follow_up = [questions[f] for f in missing if f in questions]
        
        return follow_up
    
    def reset(self):
        """Reset conversation state for new diagnosis"""
        self.conversation_state.reset()


# ============================================================================
# STEP 5: DIAGNOSTIC SESSION
# ============================================================================

class DiagnosticSession:
    """
    High-level diagnostic session management
    """
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialize diagnostic session
        
        Parameters:
        -----------
        model_dir : str
            Directory with trained model
        """
        
        self.inference = NetworkTroubleshootingInference(model_dir)
    
    def start_diagnosis(self, symptom_text: str) -> Tuple[Dict, List[str]]:
        """
        Start diagnostic session with user symptom
        
        Parameters:
        -----------
        symptom_text : str
            User's description of network problem
        
        Returns:
        --------
        Tuple : (extracted_features, follow_up_questions)
        """
        
        # Process user input
        features = self.inference.process_user_input(symptom_text)
        
        # Get follow-up questions for missing features
        follow_up = self.inference.get_follow_up_questions()
        
        # Add conversation message
        self.inference.conversation_state.add_message('user', symptom_text)
        
        return features, follow_up
    
    def answer_follow_up(self, responses: Dict) -> Tuple[str, float, List[str]]:
        """
        Process follow-up answers and generate diagnosis
        
        Parameters:
        -----------
        responses : Dict
            Answers to follow-up questions
        
        Returns:
        --------
        Tuple : (diagnosis, confidence, solutions)
        """
        
        # Fill missing features from responses
        self.inference.fill_missing_features(responses)
        
        # Generate prediction
        diagnosis, confidence, solutions = self.inference.predict()
        
        # Add bot message
        self.inference.conversation_state.add_message(
            'bot', 
            f"Diagnosis: {diagnosis} (Confidence: {confidence:.2%})"
        )
        
        return diagnosis, confidence, solutions
    
    def get_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.inference.conversation_state.history
    
    def reset(self):
        """Reset session for new diagnosis"""
        self.inference.reset()


# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("NETWORK TROUBLESHOOTING INFERENCE TEST")
    print("=" * 70)
    
    # Initialize session
    session = DiagnosticSession(model_dir='models')
    
    # Example 1: DNS Issue
    print("\n[Example 1] DNS Issue Diagnosis")
    print("-" * 70)
    
    symptom = "Internet is down but WiFi connected. Cannot open websites but ping to IP works."
    features, follow_up = session.start_diagnosis(symptom)
    
    print(f"\nUser symptom: {symptom}")
    print(f"\nExtracted features:")
    for k, v in features.items():
        print(f"  {k:20s}: {v}")
    
    print(f"\nFollow-up questions needed:")
    for i, q in enumerate(follow_up, 1):
        print(f"  {i}. {q}")
    
    # Answer follow-up
    responses = {
        'recently_updated': 0,
        'vpn_enabled': 0,
        'firewall_enabled': 1
    }
    
    print(f"\nAnswering follow-up questions...")
    diagnosis, confidence, solutions = session.answer_follow_up(responses)
    
    print(f"\n✓ DIAGNOSIS: {diagnosis}")
    print(f"✓ Confidence: {confidence:.2%}")
    print(f"\nRecommended Solutions:")
    for i, solution in enumerate(solutions, 1):
        print(f"  {i}. {solution}")
    
    print("\n" + "=" * 70)
    print("✓ Inference test completed!")
