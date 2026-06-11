import { useEffect, useState, useCallback } from 'react';
import { List, Button, Popconfirm, Typography, Spin, Empty } from 'antd';
import { PlusOutlined, DeleteOutlined, MessageOutlined, FolderOutlined } from '@ant-design/icons';
import { listSessions, deleteSession } from '../api/sessions';
import dayjs from 'dayjs';

export default function SessionList({ kbId, activeId, onSelect, onCreateClick, refreshKey = 0 }) {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      setLoading(true);
      const data = await listSessions(kbId);
      if (Array.isArray(data)) {
        setSessions(data);
      } else {
        setSessions(data.sessions || []);
      }
    } catch {
      setSessions([]);
    } finally {
      setLoading(false);
    }
  }, [kbId]);

  useEffect(() => { refresh(); }, [refresh, refreshKey]);

  const handleDelete = async (sessionId) => {
    try {
      await deleteSession(sessionId);
      if (activeId === sessionId) onSelect(null, null);
      refresh();
    } catch { /* ignore */ }
  };

  return (
    <div style={{ padding: '0 8px' }}>
      <Button
        type="dashed" block icon={<PlusOutlined />}
        onClick={onCreateClick}
        style={{ marginBottom: 8 }}
      >
        新建会话
      </Button>
      {loading ? (
        <div style={{ textAlign: 'center', padding: 24 }}><Spin /></div>
      ) : sessions.length === 0 ? (
        <Empty description="暂无会话，请点击上方按钮新建" image={Empty.PRESENTED_IMAGE_SIMPLE} />
      ) : (
        <List
          dataSource={sessions}
          renderItem={(s) => (
            <List.Item
              onClick={() => onSelect(s.session_id, s.kb_id)}
              style={{
                cursor: 'pointer', padding: '8px 12px', borderRadius: 6,
                marginBottom: 4,
                background: activeId === s.session_id ? 'rgba(22,119,255,0.08)' : 'transparent',
              }}
              actions={[
                <Popconfirm
                  key="del"
                  title="确定删除此会话？"
                  onConfirm={() => handleDelete(s.session_id)}
                >
                  <Button
                    type="text" size="small" danger
                    icon={<DeleteOutlined />}
                    onClick={(e) => e.stopPropagation()}
                  />
                </Popconfirm>,
              ]}
            >
              <List.Item.Meta
                avatar={<MessageOutlined style={{ fontSize: 18, color: '#1677ff' }} />}
                title={<Typography.Text ellipsis style={{ fontSize: 14 }}>{s.title}</Typography.Text>}
                description={
                  <span style={{ fontSize: 12, color: '#999' }}>
                    {s.kb_name ? <><FolderOutlined style={{ marginRight: 2 }} />{s.kb_name}<span style={{ margin: '0 6px' }}>·</span></> : null}
                    {dayjs(s.created_at).format('MM-DD HH:mm')}
                  </span>
                }
              />
            </List.Item>
          )}
          style={{ maxHeight: 'calc(100vh - 140px)', overflow: 'auto' }}
        />
      )}
    </div>
  );
}
