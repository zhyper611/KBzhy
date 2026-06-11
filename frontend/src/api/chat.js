const BASE = '/api';

export async function singleChat({ question, kb_id, top_k = 5, temperature = 0.5, chain_type = 'stuff', rerank_method = 'model', similarity_threshold = 0.35, enable_expansion = false, enable_rewrite = false }) {
  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, kb_id, top_k, temperature, chain_type, rerank_method, similarity_threshold, enable_expansion, enable_rewrite }),
  });
  if (!res.ok) throw new Error('对话请求失败');
  return res.json();
}

export async function multiChat(sessionId, { question, kb_id, top_k = 5, temperature = 0.5, chain_type = 'stuff', rerank_method = 'model', similarity_threshold = 0.35, enable_expansion = false, enable_rewrite = false }) {
  const res = await fetch(`${BASE}/chat/${sessionId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, kb_id, top_k, temperature, chain_type, rerank_method, similarity_threshold, enable_expansion, enable_rewrite }),
  });
  if (!res.ok) throw new Error('对话请求失败');
  return res.json();
}

export function streamChat(sessionId, { question, kb_id, top_k = 5, temperature = 0.5, chain_type = 'stuff', rerank_method = 'model', similarity_threshold = 0.35, enable_expansion = false, enable_rewrite = false }, { onToken, onDone, onError, onSources, onStatus, signal }) {
  const url = sessionId ? `${BASE}/chat/${sessionId}/stream` : `${BASE}/chat/stream`;

  fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, kb_id, top_k, temperature, chain_type, rerank_method, similarity_threshold, enable_expansion, enable_rewrite }),
    signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || '流式请求失败');
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      const handleEvent = (event) => {
        const dataLines = event
          .split(/\r?\n/)
          .filter((line) => line.startsWith('data:'))
          .map((line) => (line.startsWith('data: ') ? line.slice(6) : line.slice(5)));
        if (dataLines.length === 0) return false;

        const data = dataLines.join('\n');
        if (data === '[DONE]') {
          onDone?.();
          return true;
        }
        if (data.startsWith('[ERROR]')) {
          onError?.(data.slice(8));
          return true;
        }
        if (data.startsWith('[STATUS]')) {
          try {
            const status = JSON.parse(data.slice(8));
            onStatus?.(status);
          } catch {}
          return false;
        }
        if (data.startsWith('[SOURCES]')) {
          try {
            const sources = JSON.parse(data.slice(9));
            onSources?.(sources);
          } catch {}
          return false;
        }
        onToken?.(data);
        return false;
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split(/\r?\n\r?\n/);
        buffer = events.pop() || '';

        for (const event of events) {
          if (handleEvent(event)) return;
        }
      }
      if (buffer.trim()) {
        if (handleEvent(buffer)) return;
      }
      onDone?.();
    })
    .catch((err) => {
      if (err.name !== 'AbortError') {
        onError?.(err.message);
      }
    });
}
