# Azure OpenAI Migration Guide

## Overview

This guide documents the migration path from OpenAI API (demo/development) to Azure OpenAI (production) for the AI-powered maintenance recommendations feature.

## Why Azure OpenAI for Production?

| Feature | OpenAI API | Azure OpenAI |
|---------|-----------|--------------|
| **Data Privacy** | Data may be logged for 30 days | Data stays in your Azure tenant |
| **Training** | May use data for model improvement | Never used for training |
| **Compliance** | Limited | SOC 2, HIPAA, ISO 27001, FedRAMP |
| **Network** | Public internet | Private endpoints, VNet integration |
| **SLA** | Best effort | Enterprise SLA (99.9%+) |
| **Support** | Community/Premium | Microsoft Enterprise Support |

## Architecture Comparison

```
DEVELOPMENT (OpenAI)
┌──────────────┐      ┌──────────────┐
│  API Server  │ ───► │  OpenAI API  │ (Public Internet)
└──────────────┘      └──────────────┘

PRODUCTION (Azure OpenAI)
┌──────────────┐      ┌──────────────────────────────────┐
│  API Server  │ ───► │  Azure OpenAI                    │
│  (Your VNet) │      │  (Private Endpoint in Your VNet) │
└──────────────┘      └──────────────────────────────────┘
                      All data stays in your Azure tenant
```

## Migration Steps

### Step 1: Azure Setup (IT Team)

1. **Create Azure OpenAI Resource**
   ```bash
   az cognitiveservices account create \
     --name pdm-openai-prod \
     --resource-group pdm-resources \
     --kind OpenAI \
     --sku S0 \
     --location eastus
   ```

2. **Deploy a Model**
   ```bash
   az cognitiveservices account deployment create \
     --name pdm-openai-prod \
     --resource-group pdm-resources \
     --deployment-name gpt-4 \
     --model-name gpt-4 \
     --model-version "0613" \
     --model-format OpenAI \
     --sku-capacity 10 \
     --sku-name Standard
   ```

3. **Get Endpoint and Key**
   - Navigate to Azure Portal → Your OpenAI Resource → Keys and Endpoint
   - Copy: Endpoint URL and Key 1

### Step 2: Environment Configuration

Change from:
```bash
# Development (.env)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
```

To:
```bash
# Production (.env)
LLM_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=https://pdm-openai-prod.openai.azure.com/
AZURE_OPENAI_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AZURE_OPENAI_DEPLOYMENT=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

### Step 3: Verify Migration

```bash
# Test the recommendation engine
python ai_recommendations.py
```

Expected output:
```
Provider: azure
Generating recommendation for test machine...

RECOMMENDATION:
{
  "priority": "HIGH",
  "action": "Replace Inner Race Bearing",
  ...
  "aiGenerated": true,
  "provider": "azure"
}
```

## Code Changes Required

**None!** The `ai_recommendations.py` module uses a provider abstraction. Switching providers only requires changing environment variables.

```python
# ai_recommendations.py - No code changes needed
# The factory function handles provider selection:

def get_llm_provider() -> LLMProvider:
    if LLM_PROVIDER == "azure":
        return AzureOpenAIProvider()  # Production
    elif LLM_PROVIDER == "ollama":
        return OllamaProvider()       # Air-gapped
    else:
        return OpenAIProvider()       # Demo
```

## Private Endpoint Setup (Maximum Security)

For production environments requiring network isolation:

1. **Create Private Endpoint**
   ```bash
   az network private-endpoint create \
     --name pdm-openai-pe \
     --resource-group pdm-resources \
     --vnet-name pdm-vnet \
     --subnet default \
     --private-connection-resource-id <openai-resource-id> \
     --group-id account \
     --connection-name pdm-openai-connection
   ```

2. **Disable Public Access**
   ```bash
   az cognitiveservices account update \
     --name pdm-openai-prod \
     --resource-group pdm-resources \
     --public-network-access Disabled
   ```

Now API calls stay entirely within your Azure VNet.

## Local/Air-Gapped Alternative (Ollama)

For facilities with no internet connectivity:

1. **Install Ollama on local server**
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull llama2:13b
   ```

2. **Configure environment**
   ```bash
   LLM_PROVIDER=ollama
   OLLAMA_HOST=http://localhost:11434
   OLLAMA_MODEL=llama2:13b
   ```

## Cost Comparison

| Provider | Cost per 1K tokens | Monthly (10K requests) |
|----------|-------------------|------------------------|
| OpenAI GPT-4o-mini | $0.15 input / $0.60 output | ~$50 |
| Azure OpenAI GPT-4 | $0.03 input / $0.06 output | ~$15 |
| Ollama (local) | Hardware only | $0 (except compute) |

## Rollback Procedure

If issues occur after migration:

```bash
# Immediate rollback to OpenAI
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-xxxxx

# Restart API server
pkill -f api_server.py
python api_server.py
```

## Testing Checklist

- [ ] Azure OpenAI resource created
- [ ] Model deployed (gpt-4 or gpt-4o)
- [ ] Environment variables set correctly
- [ ] API server restarted
- [ ] Test recommendation generated successfully
- [ ] Frontend displays AI recommendations
- [ ] Fallback to rule-based works if API fails
- [ ] Response times acceptable (<3s)
- [ ] Logs show "provider": "azure"

## Support

For Azure OpenAI issues:
- Azure Support: https://azure.microsoft.com/support/
- OpenAI on Azure Docs: https://learn.microsoft.com/azure/ai-services/openai/

For application issues:
- Check logs: `tail -f /tmp/api_server.log`
- Test directly: `python ai_recommendations.py`
