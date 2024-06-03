LLM Inference on CREATE
===

How to use the AI Hub example to run inference on CREATE.

```python
ai = AI()

# One off inference.
print(ai.ask("What do cats eat?"))

# Chat with history.
print(ai.chat("What do cats eat?"))
print(ai.chat("Why do they eat that?"))
print(ai.chat("Do dogs like that too?"))

# Query an image.
with open("cat.jpg", "rb") as f:
    print(ai.ask("What is this a picture of?", images=[f.read()]))
```
