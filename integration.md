### Start the server

```bash
uvicorn fastapi_app:app --reload
```

Here are several ways you can ask a question to your FastAPI backend:

---

### 1. **Using `curl` from the command line**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the 4 steps of the TAR?", "llm_model": "qwen3"}'
```

- You can omit `"llm_model": "qwen3"` if you want to use the default model.

---

### 2. **Using Python (`requests` library)**

```python
import requests

url = "http://localhost:8000/ask"
payload = {
    "question": "What are the 4 steps of the TAR?",
    "llm_model": "qwen3"  # Optional
}
response = requests.post(url, json=payload)
print(response.json())
```

---

### 3. **Using HTTPie (a user-friendly CLI tool)**

```bash
http POST http://localhost:8000/ask question="What are the 4 steps of the TAR?" llm_model="qwen3"
```

---

### 4. **Using Swagger UI**

- Visit [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.
- Use the interactive form to send a POST request to `/ask`.

---

**Response Example:**
```json
{
  "answer": "The four steps of the TAR are: ..."
}
```

Let me know if you want a more advanced example or a ready-to-use script!