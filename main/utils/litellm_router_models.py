# Source: https://docs.litellm.ai/docs/routing
# For LiteLLM parameters, 
# see: https://docs.litellm.ai/docs/completion/input, https://docs.litellm.ai/docs/providers/bedrock
from .api_keys import OPENAI_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME

MODEL_LIST = [
    { 
        "model_name": "openai/gpt-4.1-2025-04-14", # model alias -> loadbalance between models with same `model_name`
        "litellm_params": { # params for litellm completion/embedding call 
            "model": "openai/gpt-4.1-2025-04-14", # actual model name
            "api_key": OPENAI_API_KEY,
            "rpm": 10000,
            "tpm": 30000000
        }
    }, 
    {
        "model_name": "openai/gpt-4.1-mini-2025-04-14",
        "litellm_params": {
            "model": "openai/gpt-4.1-mini-2025-04-14",
            "api_key": OPENAI_API_KEY,
            "rpm": 30000,
            "tpm": 150000000
        }
    },
    {
        "model_name": "openai/gpt-4.1-nano-2025-04-14",
        "litellm_params": {
            "model": "openai/gpt-4.1-nano-2025-04-14",
            "api_key": OPENAI_API_KEY,
            "rpm": 30000,
            "tpm": 150000000
            
        }
    },
    # Local models, deployed with vLLM server
    {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "litellm_params": {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "api_base": ""
        }
    },
    {
        "model_name": "meta-llama/Llama-3.1-70B-Instruct",
        "litellm_params": {
            "model": "meta-llama/Llama-3.1-70B-Instruct",
            "api_base": ""
        }
    },
    {
        "model_name": "Qwen/Qwen3-4B",
        "litellm_params": {
            "model": "Qwen/Qwen3-4B",
            "api_base": ""
        }
    },
    {
        "model_name": "Qwen/Qwen3-8B",
        "litellm_params": {
            "model": "Qwen/Qwen3-8B",
            "api_base": ""
        }
    },
    {
        "model_name": "Qwen/Qwen3-14B",
        "litellm_params": {
            "model": "Qwen/Qwen3-14B",
            "api_base": ""
        }
    },
    {
        "model_name": "Qwen/Qwen3-32B",
        "litellm_params": {
            "model": "Qwen/Qwen3-32B",
            "api_base": ""
        }
    },
    {
        "model_name": "intfloat/e5-mistral-7b-instruct",
        "litellm_params": {
            "model": "hosted_vllm/intfloat/e5-mistral-7b-instruct",
            "api_base": ""
        }
    },
    # AWS Bedrock models, requires AWS credentials
    {
        "model_name": "bedrock/us.amazon.nova-micro-v1:0",
        "litellm_params": {
            "model": "bedrock/us.amazon.nova-micro-v1:0",
            "aws_access_key_id": AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
            "aws_region_name": AWS_REGION_NAME,
            "rpm": 2000,
            "tpm": 4000000
        }
    },
    {
        "model_name": "bedrock/us.amazon.nova-lite-v1:0",
        "litellm_params": {
            "model": "bedrock/us.amazon.nova-lite-v1:0",
            "aws_access_key_id": AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
            "aws_region_name": AWS_REGION_NAME,
            "rpm": 2000,
            "tpm": 4000000

        }
    },
    {
        "model_name": "bedrock/us.amazon.nova-pro-v1:0",
        "litellm_params": {
            "model": "bedrock/us.amazon.nova-pro-v1:0",
            "aws_access_key_id": AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
            "aws_region_name": AWS_REGION_NAME,
            "rpm": 200,
            "tpm": 800000
        }
    },
]