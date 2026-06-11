const BASE = '/api';

// ── 知识库 CRUD ────────────────────────────────

export async function createKnowledgeBase(name) {
  const res = await fetch(`${BASE}/knowledge-bases`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || '创建知识库失败');
  }
  return res.json();
}

export async function listKnowledgeBases() {
  const res = await fetch(`${BASE}/knowledge-bases`);
  if (!res.ok) throw new Error('获取知识库列表失败');
  return res.json();
}

export async function deleteKnowledgeBase(kbId) {
  const res = await fetch(`${BASE}/knowledge-bases/${kbId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('删除知识库失败');
  return res.json();
}

// ── 文档管理（按知识库） ──────────────────────────

export async function uploadDocument(kbId, file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${BASE}/knowledge-bases/${kbId}/documents/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || '上传失败');
  }
  return res.json();
}

export async function listDocuments(kbId, { page = 1, page_size = 20, status } = {}) {
  const params = new URLSearchParams({ page, page_size });
  if (status) params.set('status', status);
  const res = await fetch(`${BASE}/knowledge-bases/${kbId}/documents?${params}`);
  if (!res.ok) throw new Error('获取文档列表失败');
  return res.json();
}

export async function getDocumentChunks(kbId, docId) {
  const res = await fetch(`${BASE}/knowledge-bases/${kbId}/documents/${docId}/chunks`);
  if (!res.ok) throw new Error('获取文档分块失败');
  return res.json();
}

export async function deleteDocument(kbId, docId) {
  const res = await fetch(`${BASE}/knowledge-bases/${kbId}/documents/${docId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('删除失败');
  return res.json();
}

export async function healthCheck() {
  const res = await fetch('/api/health');
  if (!res.ok) throw new Error('服务不可用');
  return res.json();
}
