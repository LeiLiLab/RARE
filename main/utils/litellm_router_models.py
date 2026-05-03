# Source: https://docs.litellm.ai/docs/routing
# For LiteLLM parameters, 
# see: https://docs.litellm.ai/docs/completion/input, https://docs.litellm.ai/docs/providers/bedrock
from .api_keys import OPENAI_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME, OPENROUTER_API_KEY

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
    {
        "model_name": "bedrock/us.anthropic.claude-3-haiku-20240307-v1:0",
        "litellm_params": {
            "model": "bedrock/us.anthropic.claude-3-haiku-20240307-v1:0",
            "aws_access_key_id": AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
            "aws_region_name": AWS_REGION_NAME,
            "rpm": 2000,
            "tpm": 4000000
        }
    },
    {
        "model_name": "hosted_vllm/Qwen/Qwen3-4B",
        "litellm_params": {
            "model": "hosted_vllm/Qwen/Qwen3-4B",
            "api_base": "http://localhost:8000",
        }
    },
    {
        "model_name": "hosted_vllm/Qwen/Qwen3-8B",
        "litellm_params": {
            "model": "hosted_vllm/Qwen/Qwen3-8B",
            "api_base": "http://localhost:9011/v1",
        }
    },
    {
        "model_name": "bedrock/converse/us.qwen.qwen3-32b-v1:0",
        "litellm_params": {
            "model": "bedrock/us.qwen.qwen3-32b-v1:0",
            "aws_access_key_id": AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
            "aws_region_name": AWS_REGION_NAME,
            "drop_params": True
        }
    },
    {
        "model_name": "bedrock/us.meta.llama3-1-8b-instruct-v1:0",
        "litellm_params": {
            "model": "bedrock/us.meta.llama3-1-8b-instruct-v1:0",
            "aws_access_key_id": AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
            "aws_region_name": AWS_REGION_NAME,
            "rpm": 1600,
            "tpm": 600000,
            "drop_params": True
        }
    },
    {
        "model_name": "bedrock/us.meta.llama3-1-70b-instruct-v1:0",
        "litellm_params": {
            "model": "bedrock/us.meta.llama3-1-70b-instruct-v1:0",
            "aws_access_key_id": AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
            "aws_region_name": AWS_REGION_NAME,
            "rpm": 400,
            "tpm": 300000,
            "drop_params": True
        }
    },
    {
        "model_name": "bedrock/us.meta.llama3-2-1b-instruct-v1:0",
        "litellm_params": {
            "model": "bedrock/us.meta.llama3-2-1b-instruct-v1:0",
            "aws_access_key_id": AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
            "aws_region_name": AWS_REGION_NAME,
            "rpm": 1600,
            "tpm": 600000,
            "drop_params": True
        }
    },
    {
        "model_name": "bedrock/us.meta.llama3-2-3b-instruct-v1:0",
        "litellm_params": {
            "model": "bedrock/us.meta.llama3-2-3b-instruct-v1:0",
            "aws_access_key_id": AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
            "aws_region_name": AWS_REGION_NAME,
            "rpm": 1600,
            "tpm": 600000,
            "drop_params": True
        }
    },
    {
        "model_name": "bedrock/us.meta.llama3-2-11b-instruct-v1:0",
        "litellm_params": {
            "model": "bedrock/us.meta.llama3-2-11b-instruct-v1:0",
            "aws_access_key_id": AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
            "aws_region_name": AWS_REGION_NAME,
            "rpm": 400,
            "tpm": 300000,
            "drop_params": True
        }
    },
    {
        "model_name": "bedrock/us.meta.llama3-2-90b-instruct-v1:0",
        "litellm_params": {
            "model": "bedrock/us.meta.llama3-2-90b-instruct-v1:0",
            "aws_access_key_id": AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
            "aws_region_name": AWS_REGION_NAME,
            "rpm": 400,
            "tpm": 300000,
            "drop_params": True
        }
    },
    {
        "model_name": "openrouter/meta-llama/llama-3.2-1b-instruct",
        "litellm_params": {
            "model": "openrouter/meta-llama/llama-3.2-1b-instruct",
            "api_key": OPENROUTER_API_KEY,
            "drop_params": True
        }
    },
    {
        "model_name": "openrouter/meta-llama/llama-3.2-3b-instruct",
        "litellm_params": {
            "model": "openrouter/meta-llama/llama-3.2-3b-instruct",
            "api_key": OPENROUTER_API_KEY,
            "drop_params": True
        }
    },
    {
        "model_name": "openrouter/meta-llama/llama-3.2-11b-vision-instruct",
        "litellm_params": {
            "model": "openrouter/meta-llama/llama-3.2-11b-vision-instruct",
            "api_key": OPENROUTER_API_KEY,
            "drop_params": True
        }
    },
    {
        "model_name": "openrouter/meta-llama/llama-3.2-90b-vision-instruct",
        "litellm_params": {
            "model": "openrouter/meta-llama/llama-3.2-90b-vision-instruct",
            "api_key": OPENROUTER_API_KEY,
            "drop_params": True
        }
    },
    {
        "model_name": "openrouter/anthropic/claude-3-haiku",
        "litellm_params": {
            "model": "openrouter/anthropic/claude-3-haiku",
            "api_key": OPENROUTER_API_KEY,
            "drop_params": True
        }
    },
    {
        "model_name": "openrouter/qwen/qwen3-14b",
        "litellm_params": {
            "model": "openrouter/qwen/qwen3-14b",
            "api_key": OPENROUTER_API_KEY,
            "drop_params": True
        }
    },
    
]