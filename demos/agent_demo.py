"""
AI Agent Demo - Reliability Engineering Assistant
==================================================
Interprets anomaly detection results and provides actionable insights.

Features:
- MOCK_MODE for demos without API keys
- LLM integration for production use
- Veteran engineer-style insights
- Formatted, executive-ready output

Author: AI/ML Engineering Team
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

# MOCK_MODE: Set to True for demos without API keys
MOCK_MODE = True

# API Configuration (for production use)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Data file
INSIGHT_REPORT_FILE = "insight_report.json"

# =============================================================================
# MOCK RESPONSE (IMPRESSIVE VETERAN ENGINEER)
# =============================================================================

def get_mock_response(anomaly_data: Dict[str, Any]) -> str:
    """
    Pre-written response from a "20-year veteran engineer".
    This is used in MOCK_MODE for demos.
    """
    return """Here's what I found after analyzing the vibration data:

At 21:01 on December 13th, we saw vibration levels spike to 0.374g—that's 47% above our rolling baseline of 0.255g. This isn't random noise. The pattern is too clean.

Looking back through the event logs, I found the smoking gun: the night cleaning crew entered Zone 3 at 20:45, exactly 16 minutes before the anomaly. This matches a classic EMI signature—they're plugging in industrial floor scrubbers and carpet extractors without proper power conditioning.

Here's why this matters: These micro-vibrations are accelerating bearing wear on the robotic arm's joint assemblies. Left unchecked, we're looking at premature failure in 6-8 weeks instead of the normal 18-month service interval. That's a $47,000 replacement plus 12 hours of line downtime.

The fix is straightforward: Install a 20-amp isolated power circuit for cleaning equipment in Zone 3, away from the production power grid. Cost is around $1,200 in materials and 4 hours of electrician time. Schedule it during the next planned maintenance window.

I've seen this exact pattern at three other plants. It's always the cleaning crew, and it's always fixable."""

# =============================================================================
# LLM INTEGRATION (PRODUCTION MODE)
# =============================================================================

def call_openai_api(system_prompt: str, user_prompt: str) -> str:
    """
    Call OpenAI API for production use.
    Requires OPENAI_API_KEY environment variable.
    """
    try:
        # Uncomment when ready for production:
        # from openai import OpenAI
        # client = OpenAI(api_key=OPENAI_API_KEY)
        # 
        # response = client.chat.completions.create(
        #     model="gpt-4",
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_prompt}
        #     ],
        #     temperature=0.7,
        #     max_tokens=500
        # )
        # return response.choices[0].message.content
        
        return "ERROR: OpenAI integration not yet enabled. Set MOCK_MODE=True or implement API calls."
    
    except Exception as e:
        return f"ERROR calling OpenAI API: {str(e)}"

def call_anthropic_api(system_prompt: str, user_prompt: str) -> str:
    """
    Call Anthropic Claude API for production use.
    Requires ANTHROPIC_API_KEY environment variable.
    """
    try:
        # Uncomment when ready for production:
        # import anthropic
        # client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        # 
        # message = client.messages.create(
        #     model="claude-3-5-sonnet-20241022",
        #     max_tokens=500,
        #     system=system_prompt,
        #     messages=[
        #         {"role": "user", "content": user_prompt}
        #     ]
        # )
        # return message.content[0].text
        
        return "ERROR: Anthropic integration not yet enabled. Set MOCK_MODE=True or implement API calls."
    
    except Exception as e:
        return f"ERROR calling Anthropic API: {str(e)}"

# =============================================================================
# DATA PROCESSING
# =============================================================================

def load_insight_report(filepath: str) -> Dict[str, Any]:
    """Load the anomaly detection insight report."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Insight report not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)

def find_most_recent_anomaly_with_cause(report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Filter anomalies to find the most recent one with a root cause.
    Returns the single best example for demo purposes.
    """
    anomalies = report.get('anomalies', [])
    
    # Filter to only anomalies with root causes
    with_causes = [a for a in anomalies if a.get('root_cause_found', False)]
    
    if not with_causes:
        return None
    
    # Sort by timestamp (most recent first)
    with_causes.sort(key=lambda x: x['anomaly_timestamp'], reverse=True)
    
    return with_causes[0]

# =============================================================================
# PROMPT CONSTRUCTION
# =============================================================================

def construct_prompts(anomaly: Dict[str, Any]) -> tuple[str, str]:
    """
    Construct the system and user prompts for the LLM.
    
    Returns:
        (system_prompt, user_prompt)
    """
    system_prompt = """You are an AI Reliability Engineer at a Tier 1 Automotive Plant. 

Your role is to interpret sensor anomalies and provide actionable insights to plant managers.

Guidelines:
- Speak concisely and directly
- Value data and precision
- Avoid unnecessary jargon
- Provide specific, actionable recommendations
- Reference actual numbers from the data
- Think like a 20-year veteran who has seen it all"""

    # Format anomaly data for readability
    anomaly_summary = f"""
Anomaly Timestamp: {anomaly['anomaly_timestamp']}
Severity: {anomaly['severity']}
Anomaly Score: {anomaly['anomaly_score']:.4f}
Vibration Reading: {anomaly['vibration_x']:.3f}g
Rolling Average: {anomaly['vibration_rolling_mean']:.3f}g
Rolling Std Dev: {anomaly['vibration_rolling_std']:.3f}g

Root Cause Identified:
- Event Type: {anomaly['event_type']}
- Event Timestamp: {anomaly['event_timestamp']}
- Staff ID: {anomaly['staff_id']}
- Time Before Anomaly: {anomaly['time_delta_minutes']} minutes
- Confidence: {anomaly['confidence']}
"""

    user_prompt = f"""Here is a detected anomaly from our predictive maintenance system:

{anomaly_summary}

Explain what happened, why it matters to our production line, and recommend a specific fix. 

Be direct and practical. The plant manager reading this doesn't have time for fluff."""

    return system_prompt, user_prompt

# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def print_formatted_alert(anomaly: Dict[str, Any], explanation: str):
    """
    Print the AI's response in an executive-ready format.
    """
    # Header
    print()
    print("=" * 80)
    print("🛑 CRITICAL ALERT - LINE 4")
    print("=" * 80)
    print()
    
    # Metadata
    print(f"📅 Detected: {anomaly['anomaly_timestamp']}")
    print(f"📊 Severity: {anomaly['severity']}")
    print(f"🎯 Root Cause: {anomaly['event_type']}")
    print()
    print("-" * 80)
    print()
    
    # AI Explanation
    print("🤖 AI RELIABILITY ENGINEER ANALYSIS:")
    print()
    print(explanation)
    print()
    
    # Footer
    print("-" * 80)
    print()
    print(f"📈 Confidence Score: {anomaly['confidence']}")
    print(f"⚡ Anomaly Severity Score: {anomaly['anomaly_score']:.4f}")
    print(f"🔧 Recommended Action: Review and implement suggested fix")
    print()
    print("=" * 80)
    print()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Execute the AI agent demo."""
    print()
    print("=" * 80)
    print("AI RELIABILITY ENGINEER - ANOMALY INTERPRETER")
    print("=" * 80)
    print()
    
    if MOCK_MODE:
        print("⚙️  Running in MOCK MODE (demo without API keys)")
    else:
        print("🌐 Running in PRODUCTION MODE (using LLM API)")
    print()
    
    try:
        # Step 1: Load insight report
        print(f"📂 Loading insight report: {INSIGHT_REPORT_FILE}")
        report = load_insight_report(INSIGHT_REPORT_FILE)
        print(f"   ✓ Loaded {report['summary']['total_anomalies']} anomalies")
        print()
        
        # Step 2: Find most recent anomaly with root cause
        print("🔍 Finding most recent anomaly with identified root cause...")
        anomaly = find_most_recent_anomaly_with_cause(report)
        
        if not anomaly:
            print("   ⚠️  No anomalies with root causes found in report.")
            return 1
        
        print(f"   ✓ Found: {anomaly['event_type']} at {anomaly['anomaly_timestamp']}")
        print()
        
        # Step 3: Construct prompts
        print("🧠 Constructing LLM prompts...")
        system_prompt, user_prompt = construct_prompts(anomaly)
        print("   ✓ Prompts ready")
        print()
        
        # Step 4: Get AI response
        print("💭 Generating AI analysis...")
        
        if MOCK_MODE:
            # Use pre-written veteran engineer response
            explanation = get_mock_response(anomaly)
            print("   ✓ Mock response generated")
        else:
            # Call actual LLM API (OpenAI by default)
            if OPENAI_API_KEY:
                explanation = call_openai_api(system_prompt, user_prompt)
            elif ANTHROPIC_API_KEY:
                explanation = call_anthropic_api(system_prompt, user_prompt)
            else:
                print("   ❌ No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
                print("   💡 Tip: Set MOCK_MODE=True for demos without API keys")
                return 1
        
        print()
        
        # Step 5: Display formatted output
        print_formatted_alert(anomaly, explanation)
        
        # Success message
        print("✅ AI Agent Demo Complete!")
        print()
        print("Next Steps:")
        print("  • Review the AI's recommendation")
        print("  • Set MOCK_MODE=False to use live LLM APIs")
        print("  • Integrate with alerting system for real-time notifications")
        print()
        
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print("   Tip: Run analytics_engine.py first to generate insight_report.json")
        return 1
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
