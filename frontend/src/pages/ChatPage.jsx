import { useState, useRef, useEffect, useCallback } from 'react';
import { Input, Button, Layout, Empty, Typography, Modal, message } from 'antd';
import { SendOutlined, PauseCircleOutlined } from '@ant-design/icons';
import ChatBubble from '../components/ChatBubble';
import SessionList from '../components/SessionList';
import ConfigPanel from '../components/ConfigPanel';
import KBSelector from '../components/KBSelector';
import { useStreamChat } from '../hooks/useStreamChat';
import { getSessionMessages, createSession } from '../api/sessions';

export default function ChatPage() {
  const [sessionId, setSessionId] = useState(null);
  const [kbId, setKbId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [params, setParams] = useState({ top_k: 5, temperature: 0.5, chain_type: 'stuff', rerank_method: 'model', similarity_threshold: 0.35, enable_expansion: false, enable_rewrite: false });
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [pendingKbId, setPendingKbId] = useState(null);
  const [creating, setCreating] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [textAreaKey, setTextAreaKey] = useState(0);

  const messagesEndRef = useRef(null);
  const assistantRef = useRef({ content: '', sources: [] });

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => { scrollToBottom(); }, [messages]);

  const { streaming, startStream, stopStream } = useStreamChat(sessionId);

  const appendAssistant = useCallback((update) => {
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last?.role === 'assistant' && last.isStreaming) {
        return [...prev.slice(0, -1), { ...last, ...update }];
      }
      return [...prev, { role: 'assistant', content: '', sources: [], isStreaming: true, ...update }];
    });
  }, []);

  const finalizeAssistant = useCallback(() => {
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (!last || last.role !== 'assistant') return prev;
      return [...prev.slice(0, -1), { ...last, isStreaming: false }];
    });
  }, []);

  const handleSend = () => {
    const q = inputValue.trim();
    if (!q || streaming) return;
    const isFirstQuestion = messages.length === 0;

    setMessages((prev) => [...prev, { role: 'user', content: q }]);
    setInputValue('');
    setTextAreaKey((k) => k + 1);

    assistantRef.current = { content: '', sources: [] };
    appendAssistant({ content: '', sources: [] });

    startStream(q, { ...params, kb_id: kbId }, {
      onStatus: (status) => {
        appendAssistant({ status });
      },
      onToken: (token) => {
        assistantRef.current.content += token;
        appendAssistant({ content: assistantRef.current.content, sources: assistantRef.current.sources });
      },
      onSources: (sources) => {
        assistantRef.current.sources = sources;
        appendAssistant({ content: assistantRef.current.content, sources });
      },
      onDone: () => {
        finalizeAssistant();
        if (isFirstQuestion) setRefreshKey((k) => k + 1);
      },
      onError: (msg) => {
        assistantRef.current.content += `\n\n> 错误: ${msg}`;
        appendAssistant({ content: assistantRef.current.content });
        finalizeAssistant();
        if (isFirstQuestion) setRefreshKey((k) => k + 1);
      },
    });
  };

  const handlePressEnter = (e) => {
    if (e.shiftKey) return;
    e.preventDefault();
    handleSend();
  };

  const handleSessionSelect = async (id, kbIdFromSession) => {
    setSessionId(id);
    if (!id) {
      setKbId(null);
      setMessages([]);
      return;
    }
    setKbId(kbIdFromSession || null);
    try {
      const data = await getSessionMessages(id);
      const history = (data.messages || []).map((m) => ({
        role: m.role,
        content: m.content,
        sources: m.sources || [],
        isStreaming: false,
      }));
      setMessages(history);
    } catch {
      setMessages([]);
    }
  };

  const handleCreateClick = () => {
    setPendingKbId(null);
    setCreateModalOpen(true);
  };

  const handleCreateConfirm = async () => {
    if (!pendingKbId) {
      message.warning('请先选择一个知识库');
      return;
    }
    setCreating(true);
    try {
      const data = await createSession('新会话', pendingKbId);
      setCreateModalOpen(false);
      setPendingKbId(null);
      handleSessionSelect(data.session_id, data.kb_id);
      setRefreshKey((k) => k + 1);
    } catch (err) {
      message.error('创建会话失败: ' + (err.message || '未知错误'));
    } finally {
      setCreating(false);
    }
  };

  const emptyHint = () => {
    if (sessionId && kbId) return '开始提问吧';
    if (sessionId && !kbId) return '会话关联的知识库不存在';
    return '请新建一个会话开始对话';
  };

  return (
    <Layout style={{ height: '100%', background: '#fff' }}>
      <Layout.Sider width={260} style={{ background: '#fafafa', borderRight: '1px solid #f0f0f0' }}>
        <div style={{ padding: '12px 0' }}>
          <Typography.Title level={5} style={{ padding: '0 16px', margin: '0 0 8px' }}>会话列表</Typography.Title>
          <SessionList
            kbId={null}
            activeId={sessionId}
            onSelect={handleSessionSelect}
            onCreateClick={handleCreateClick}
            refreshKey={refreshKey}
          />
        </div>
      </Layout.Sider>
      <Layout.Content style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        <div
          className="chat-messages"
          style={{ flex: 1, overflow: 'auto', padding: '16px 24px' }}
        >
          {messages.length === 0 ? (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
              <Empty description={emptyHint()} />
            </div>
          ) : (
            messages.map((m, i) => (
              <ChatBubble key={i} message={m} />
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        <div style={{
          padding: '12px 24px 16px',
          borderTop: '1px solid #f0f0f0',
          background: '#fafafa',
        }}>
          <Input.TextArea
            key={textAreaKey}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onPressEnter={handlePressEnter}
            placeholder="输入问题，Enter 发送，Shift+Enter 换行"
            autoSize={{ minRows: 1, maxRows: 4 }}
            disabled={!sessionId || !kbId}
            style={{ marginBottom: 8 }}
          />
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
            {streaming && (
              <Button icon={<PauseCircleOutlined />} onClick={stopStream}>停止生成</Button>
            )}
            <Button
              type="primary"
              icon={<SendOutlined />}
              onClick={handleSend}
              loading={streaming}
              disabled={!sessionId || !kbId || !inputValue.trim()}
            >
              发送
            </Button>
          </div>
        </div>
      </Layout.Content>
      <Layout.Sider width={280} style={{ background: '#fafafa', borderLeft: '1px solid #f0f0f0' }}>
        <ConfigPanel values={params} onChange={setParams} />
      </Layout.Sider>

      <Modal
        title="新建会话"
        open={createModalOpen}
        onOk={handleCreateConfirm}
        onCancel={() => { setCreateModalOpen(false); setPendingKbId(null); }}
        okText="创建"
        cancelText="取消"
        confirmLoading={creating}
        okButtonProps={{ disabled: !pendingKbId }}
      >
        <div style={{ marginBottom: 8 }}>
          <Typography.Text type="secondary">请选择知识库：</Typography.Text>
        </div>
        <KBSelector value={pendingKbId} onChange={setPendingKbId} disabled={false} allowCreate={false} />
      </Modal>
    </Layout>
  );
}
