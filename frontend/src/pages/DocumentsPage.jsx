import { useState, useEffect, useCallback } from 'react';
import { Typography, Tag, Card, Row, Col, Button, Popconfirm, Modal, Input, Empty, Spin, message } from 'antd';
import {
  CloudServerOutlined, PlusOutlined, DatabaseOutlined,
  DeleteOutlined, ArrowLeftOutlined, FileTextOutlined,
} from '@ant-design/icons';
import UploadZone from '../components/UploadZone';
import DocumentTable from '../components/DocumentTable';
import {
  healthCheck, listKnowledgeBases, createKnowledgeBase, deleteKnowledgeBase,
} from '../api/documents';
import dayjs from 'dayjs';

export default function DocumentsPage() {
  const [selectedKb, setSelectedKb] = useState(null); // kb_id or null
  const [kbs, setKbs] = useState([]);
  const [kbLoading, setKbLoading] = useState(true);
  const [refreshKey, setRefreshKey] = useState(0);
  const [health, setHealth] = useState(null);

  // create modal
  const [createOpen, setCreateOpen] = useState(false);
  const [newName, setNewName] = useState('');
  const [creating, setCreating] = useState(false);

  const checkHealth = useCallback(async () => {
    try {
      const data = await healthCheck();
      setHealth(data);
    } catch {
      setHealth({ status: 'error' });
    }
  }, []);

  const refreshKBs = useCallback(async () => {
    try {
      setKbLoading(true);
      const data = await listKnowledgeBases();
      setKbs(data || []);
    } catch {
      setKbs([]);
    } finally {
      setKbLoading(false);
    }
  }, []);

  useEffect(() => { checkHealth(); }, [checkHealth]);
  useEffect(() => { refreshKBs(); }, [refreshKBs]);

  const handleUploaded = () => {
    setRefreshKey((k) => k + 1);
    refreshKBs(); // update doc count on kb cards
  };

  const handleCreate = async () => {
    if (!newName.trim()) return;
    setCreating(true);
    try {
      const kb = await createKnowledgeBase(newName.trim());
      message.success(`知识库「${newName.trim()}」已创建`);
      setNewName('');
      setCreateOpen(false);
      await refreshKBs();
      setSelectedKb(kb.kb_id);
    } catch (err) {
      message.error(err.message);
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (kbId) => {
    try {
      const deleted = kbs.find((kb) => kb.kb_id === kbId);
      await deleteKnowledgeBase(kbId);
      message.success(`知识库「${deleted?.name || kbId}」已删除`);
      if (selectedKb === kbId) setSelectedKb(null);
      await refreshKBs();
    } catch (err) {
      message.error(err.message);
    }
  };

  const handleBack = () => {
    setSelectedKb(null);
    refreshKBs();
  };

  const currentKb = kbs.find((kb) => kb.kb_id === selectedKb);

  // ── KB detail view ─────────────────────────────
  if (selectedKb) {
    return (
      <div style={{ padding: 24, height: '100%', overflow: 'auto' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
          <Button icon={<ArrowLeftOutlined />} onClick={handleBack}>返回</Button>
          <Typography.Title level={4} style={{ margin: 0 }}>
            <DatabaseOutlined style={{ marginRight: 8 }} />
            {currentKb?.name || selectedKb}
          </Typography.Title>
          <Tag>{currentKb?.doc_count || 0} 个文档</Tag>
        </div>
        <UploadZone kbId={selectedKb} onUploaded={handleUploaded} />
        <DocumentTable kbId={selectedKb} refreshKey={refreshKey} />
      </div>
    );
  }

  // ── KB list view ───────────────────────────────
  return (
    <div style={{ padding: 24, height: '100%', overflow: 'auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <Typography.Title level={4} style={{ margin: 0 }}>文档管理</Typography.Title>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateOpen(true)}>
            新建知识库
          </Button>
          <Tag
            icon={<CloudServerOutlined />}
            color={health?.status === 'healthy' ? 'success' : health?.status === 'error' ? 'error' : 'processing'}
          >
            {health?.status === 'healthy' ? '服务正常' : health?.status === 'error' ? '服务异常' : '检查中...'}
          </Tag>
        </div>
      </div>

      {kbLoading ? (
        <div style={{ textAlign: 'center', padding: 64 }}><Spin size="large" /></div>
      ) : kbs.length === 0 ? (
        <Empty description="暂无知识库，请新建" style={{ marginTop: 64 }} />
      ) : (
        <Row gutter={[16, 16]}>
          {kbs.map((kb) => (
            <Col key={kb.kb_id} xs={24} sm={12} md={8} lg={6}>
              <Card
                hoverable
                onClick={() => setSelectedKb(kb.kb_id)}
                actions={[
                  <Popconfirm
                    key="del"
                    title="确定删除此知识库？"
                    description="将同时删除所有文档和向量数据，不可恢复。"
                    onConfirm={(e) => { e.stopPropagation(); handleDelete(kb.kb_id); }}
                    onCancel={(e) => e.stopPropagation()}
                    okText="确定删除"
                    cancelText="取消"
                    okButtonProps={{ danger: true }}
                  >
                    <Button
                      type="text" danger size="small" icon={<DeleteOutlined />}
                      onClick={(e) => e.stopPropagation()}
                    >
                      删除
                    </Button>
                  </Popconfirm>,
                ]}
              >
                <Card.Meta
                  avatar={<DatabaseOutlined style={{ fontSize: 32, color: '#1677ff' }} />}
                  title={kb.name}
                  description={
                    <div>
                      <div style={{ marginBottom: 4 }}>
                        <FileTextOutlined style={{ marginRight: 4 }} />
                        {kb.doc_count} 个文档
                      </div>
                      {kb.created_at && (
                        <div style={{ fontSize: 12, color: '#999' }}>
                          创建于 {dayjs(kb.created_at).format('YYYY-MM-DD HH:mm')}
                        </div>
                      )}
                    </div>
                  }
                />
              </Card>
            </Col>
          ))}
        </Row>
      )}

      <Modal
        title="新建知识库"
        open={createOpen}
        onOk={handleCreate}
        onCancel={() => { setCreateOpen(false); setNewName(''); }}
        confirmLoading={creating}
        okText="创建"
        cancelText="取消"
      >
        <Input
          placeholder="请输入知识库名称"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          onPressEnter={handleCreate}
          maxLength={100}
        />
      </Modal>
    </div>
  );
}
