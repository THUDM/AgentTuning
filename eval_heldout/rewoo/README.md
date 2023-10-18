# ReWOO

This evaluation code is adapted from [ReWOO](https://github.com/billxbf/ReWOO/tree/main).

## Installation 🔧

```
pip install -r requirements.txt
```

Generate API keys from [OpenAI](https://openai.com/blog/openai-api), [WolframAlpha](https://products.wolframalpha.com/api) and [Serper](https://serper.dev). Then save the keys to `./keys/openai.key`, `./keys/wolfram.key` and `./keys/serper.key` respectively.

You need to provide OpenAI key even if you are evaluating local-hosted models, because the evaluator relies on it.

## Evaluation

### GPT

```bash
bash eval-gpt-3.5-turbo.sh
```

or 

```bash
bash eval-gpt-4.sh
```

### TGI (Text Generation Inference)

```bash
bash eval-tgi.sh
```

---

The results will be saved to `logs` and `results` directory. After evaluation is done, you can run `python metrics.py` to get the metrics.