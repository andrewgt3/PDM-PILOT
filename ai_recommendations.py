"""
AI-Powered Maintenance Recommendation Engine

This module provides intelligent maintenance recommendations using LLM APIs.
Supports multiple providers with easy switching between:
- OpenAI (demo/development)
- Azure OpenAI (production - data stays in your tenant)
- Local Ollama (air-gapped/maximum privacy)

For production deployment, see: docs/AZURE_OPENAI_MIGRATION.md
"""

import os
import json
from typing import Dict, Optional
from abc import ABC, abstractmethod


# =============================================================================
# PROVIDER CONFIGURATION
# =============================================================================
# Set this environment variable to switch providers:
#   - "openai" (default for demo)
#   - "azure" (production)
#   - "ollama" (local/private)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Azure OpenAI Configuration (for production)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# Ollama Configuration (local)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2:13b")


# =============================================================================
# SYSTEM PROMPT - Maintenance Expert Context
# =============================================================================
MAINTENANCE_SYSTEM_PROMPT = """You are an expert predictive maintenance engineer specializing in automotive manufacturing equipment. Your role is to analyze sensor data and provide actionable maintenance recommendations.

RESPONSE FORMAT - Always respond with valid JSON:
{
    "priority": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW",
    "action": "Brief action title (e.g., 'Replace Inner Race Bearing')",
    "reasoning": "2-3 sentences explaining why based on the sensor data",
    "timeWindow": "Specific time recommendation (e.g., 'Within 24 hours', 'Schedule for next planned downtime')",
    "parts": ["List", "of", "required", "parts"],
    "estimatedDowntime": "Duration estimate (e.g., '2-4 hours')",
    "safetyNotes": "Any safety considerations or lockout/tagout requirements"
}

GUIDELINES:
1. Base recommendations on the actual sensor values provided
2. Consider bearing fault frequencies (BPFO = outer race, BPFI = inner race, BSF = ball, FTF = cage)
3. Degradation > 70% is critical, > 50% is warning
4. Consider equipment type when recommending actions
5. Be specific about parts and procedures
6. Always mention safety for high-energy equipment (presses, robots)"""


# =============================================================================
# ABSTRACT PROVIDER INTERFACE
# =============================================================================
class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_recommendation(self, machine_context: str) -> Dict:
        """Generate maintenance recommendation from machine context."""
        pass


# =============================================================================
# OPENAI PROVIDER (Demo/Development)
# =============================================================================
class OpenAIProvider(LLMProvider):
    """OpenAI API provider for demo/development use."""
    
    def __init__(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.available = bool(OPENAI_API_KEY)
        except ImportError:
            self.client = None
            self.available = False
            print("[WARNING] openai package not installed. Run: pip install openai")
    
    def generate_recommendation(self, machine_context: str) -> Dict:
        if not self.available:
            return self._fallback_response("OpenAI API not configured")
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": MAINTENANCE_SYSTEM_PROMPT},
                    {"role": "user", "content": machine_context}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,  # Lower for more consistent recommendations
                max_tokens=500
            )
            
            return json.loads(response.choices[0].message.content)
        
        except Exception as e:
            print(f"[OpenAI Error] {e}")
            return self._fallback_response(str(e))
    
    def _fallback_response(self, error: str) -> Dict:
        return {
            "priority": "MEDIUM",
            "action": "Schedule Routine Inspection",
            "reasoning": f"AI recommendation unavailable ({error}). Defaulting to routine inspection based on sensor data patterns.",
            "timeWindow": "Within 7 days",
            "parts": ["Inspection kit"],
            "estimatedDowntime": "1-2 hours",
            "safetyNotes": "Follow standard lockout/tagout procedures",
            "aiGenerated": False
        }


# =============================================================================
# AZURE OPENAI PROVIDER (Production)
# =============================================================================
class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI provider for production use with data privacy."""
    
    def __init__(self):
        try:
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_KEY,
                api_version=AZURE_OPENAI_API_VERSION
            )
            self.available = bool(AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY)
        except ImportError:
            self.client = None
            self.available = False
    
    def generate_recommendation(self, machine_context: str) -> Dict:
        if not self.available:
            return self._fallback_response("Azure OpenAI not configured")
        
        try:
            response = self.client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,  # Azure uses deployment name, not model
                messages=[
                    {"role": "system", "content": MAINTENANCE_SYSTEM_PROMPT},
                    {"role": "user", "content": machine_context}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content)
            result["aiGenerated"] = True
            result["provider"] = "azure"
            return result
        
        except Exception as e:
            print(f"[Azure OpenAI Error] {e}")
            return self._fallback_response(str(e))
    
    def _fallback_response(self, error: str) -> Dict:
        return {
            "priority": "MEDIUM",
            "action": "Schedule Routine Inspection",
            "reasoning": f"AI recommendation unavailable ({error}). Defaulting to routine inspection.",
            "timeWindow": "Within 7 days",
            "parts": ["Inspection kit"],
            "estimatedDowntime": "1-2 hours",
            "safetyNotes": "Follow standard lockout/tagout procedures",
            "aiGenerated": False
        }


# =============================================================================
# OLLAMA PROVIDER (Local/Private)
# =============================================================================
class OllamaProvider(LLMProvider):
    """Local Ollama provider for maximum privacy."""
    
    def __init__(self):
        import requests
        self.requests = requests
        self.available = True  # Assume available, will fail gracefully
    
    def generate_recommendation(self, machine_context: str) -> Dict:
        try:
            response = self.requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": f"{MAINTENANCE_SYSTEM_PROMPT}\n\nAnalyze this machine:\n{machine_context}\n\nRespond with JSON only:",
                    "stream": False,
                    "format": "json"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = json.loads(response.json()["response"])
                result["aiGenerated"] = True
                result["provider"] = "ollama"
                return result
            else:
                return self._fallback_response(f"Ollama returned {response.status_code}")
        
        except Exception as e:
            print(f"[Ollama Error] {e}")
            return self._fallback_response(str(e))
    
    def _fallback_response(self, error: str) -> Dict:
        return {
            "priority": "MEDIUM",
            "action": "Schedule Routine Inspection",
            "reasoning": f"Local AI unavailable ({error}). Using fallback recommendation.",
            "timeWindow": "Within 7 days",
            "parts": ["Inspection kit"],
            "estimatedDowntime": "1-2 hours",
            "safetyNotes": "Follow standard lockout/tagout procedures",
            "aiGenerated": False
        }


# =============================================================================
# FACTORY FUNCTION - Get Configured Provider
# =============================================================================
def get_llm_provider() -> LLMProvider:
    """Get the configured LLM provider based on environment."""
    if LLM_PROVIDER == "azure":
        return AzureOpenAIProvider()
    elif LLM_PROVIDER == "ollama":
        return OllamaProvider()
    else:  # Default to OpenAI
        return OpenAIProvider()


# =============================================================================
# MAIN RECOMMENDATION FUNCTION
# =============================================================================
def generate_maintenance_recommendation(
    machine_id: str,
    machine_name: str,
    equipment_type: str,
    sensor_data: Dict,
    shop: str = "Unknown",
    line: str = "Unknown"
) -> Dict:
    """
    Generate AI-powered maintenance recommendation for a machine.
    
    Args:
        machine_id: Equipment identifier (e.g., "WB-001")
        machine_name: Human-readable name (e.g., "6-Axis Welder #1")
        equipment_type: Type of equipment (e.g., "Spot Welder")
        sensor_data: Dictionary containing sensor readings
        shop: Plant area (e.g., "Body Shop")
        line: Production line (e.g., "Underbody Weld Cell")
    
    Returns:
        Dictionary with recommendation details
    """
    # Build context string for LLM
    context = f"""
MACHINE ANALYSIS REQUEST
========================
Machine ID: {machine_id}
Name: {machine_name}
Type: {equipment_type}
Location: {shop} > {line}

SENSOR READINGS:
- Failure Probability: {sensor_data.get('failure_probability', 0) * 100:.1f}%
- Degradation Score: {sensor_data.get('degradation_score', 0) * 100:.1f}%
- BPFI Amplitude (Inner Race): {sensor_data.get('bpfi_amp', 0):.4f} g
- BPFO Amplitude (Outer Race): {sensor_data.get('bpfo_amp', 0):.4f} g
- BSF Amplitude (Ball Spin): {sensor_data.get('bsf_amp', 0):.4f} g
- FTF Amplitude (Cage): {sensor_data.get('ftf_amp', 0):.4f} g
- Rotational Speed: {sensor_data.get('rotational_speed', 0):.0f} RPM
- Temperature: {sensor_data.get('temperature', 0):.1f}°C
- RUL (Remaining Useful Life): {sensor_data.get('rul_days', 'N/A')} days

Please analyze this equipment and provide your maintenance recommendation.
"""
    
    # Get provider and generate recommendation
    provider = get_llm_provider()
    recommendation = provider.generate_recommendation(context)
    
    # Add metadata
    recommendation["machine_id"] = machine_id
    recommendation["timestamp"] = __import__("datetime").datetime.now().isoformat()
    recommendation["llmProvider"] = LLM_PROVIDER
    
    return recommendation


# =============================================================================
# DEMO / TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("AI MAINTENANCE RECOMMENDATION ENGINE")
    print("=" * 60)
    print(f"Provider: {LLM_PROVIDER}")
    print()
    
    # Test with sample data
    test_data = {
        "failure_probability": 0.75,
        "degradation_score": 0.68,
        "bpfi_amp": 0.42,
        "bpfo_amp": 0.15,
        "bsf_amp": 0.08,
        "ftf_amp": 0.05,
        "rotational_speed": 1800,
        "temperature": 78.5,
        "rul_days": 12
    }
    
    print("Generating recommendation for test machine...")
    recommendation = generate_maintenance_recommendation(
        machine_id="WB-001",
        machine_name="6-Axis Welder #1",
        equipment_type="Spot Welder",
        sensor_data=test_data,
        shop="Body Shop",
        line="Underbody Weld Cell"
    )
    
    print("\nRECOMMENDATION:")
    print(json.dumps(recommendation, indent=2))
