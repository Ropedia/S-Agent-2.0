# LiteLLM / vLLM Scripts


## Files



## Quick Start

```bash
conda create -n litellm_1.83.9 python=3.13
conda activate litellm_1.83.9
pip install "litellm[proxy]==1.83.9"
cp scripts/serve_llm/litellm_config.example.yaml scripts/serve_llm/litellm_config.yaml
bash scripts/serve_llm/serve_litellm.sh
```


- Custom agent runtime：`../../docs_dev/custom-agents.md`
