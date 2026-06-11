const BASE = '/api';

export async function createSession(title = '新会话', kbId = null) {
  const res = await fetch(`${BASE}/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, kb_id: kbId }),
  });
  if (!res.ok) throw new Error('创建会话失败');
  return res.json();
}

export async function listSessions(kbId = null) {
  const url = kbId ? `${BASE}/sessions?kb_id=${encodeURIComponent(kbId)}` : `${BASE}/sessions`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('获取会话列表失败');
  return res.json();
}

export async function getSessionMessages(sessionId) {
  const res = await fetch(`${BASE}/sessions/${sessionId}/messages`);
  if (!res.ok) throw new Error('获取会话历史失败');
  return res.json();
}

export async function deleteSession(sessionId) {
  const res = await fetch(`${BASE}/sessions/${sessionId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('删除会话失败');
  return res.json();
}
