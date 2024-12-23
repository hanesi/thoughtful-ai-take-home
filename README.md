# thoughtful-ai-take-home

install text generation

```
docker pull --platform linux/amd64 ghcr.io/huggingface/text-generation-inference:latest
```

start the server

```
docker run --platform linux/amd64 --rm -p 8080:80 ghcr.io/huggingface/text-generation-inference:latest --model-id gpt2

```

run the bot

```
python chatbot.py
```