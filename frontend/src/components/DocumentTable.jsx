import { useEffect, useState, useCallback } from 'react';
import { Table, Tag, Button, Popconfirm, Typography, Drawer, List, Spin, Empty, Space, message } from 'antd';
import { DeleteOutlined, ReloadOutlined, EyeOutlined } from '@ant-design/icons';
import { listDocuments, deleteDocument, getDocumentChunks } from '../api/documents';
import dayjs from 'dayjs';

const STATUS_MAP = {
  queued: { color: 'default', text: '排队中' },
  parsing: { color: 'processing', text: '解析中' },
  chunking: { color: 'processing', text: '分块中' },
  indexing: { color: 'processing', text: '索引中' },
  ready: { color: 'success', text: '就绪' },
  failed: { color: 'error', text: '失败' },
  deleting: { color: 'warning', text: '删除中' },
  uploaded: { color: 'default', text: '已上传' },
};

const ACTIVE_STATUSES = new Set(['queued', 'parsing', 'chunking', 'indexing', 'deleting']);

export default function DocumentTable({ kbId, refreshKey }) {
  const [docs, setDocs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [chunks, setChunks] = useState([]);
  const [chunksLoading, setChunksLoading] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const refresh = useCallback(async () => {
    if (!kbId) return;
    try {
      setLoading(true);
      const data = await listDocuments(kbId, { page_size: 100 });
      setDocs(data.documents || []);
    } catch {
      setDocs([]);
    } finally {
      setLoading(false);
    }
  }, [kbId]);

  useEffect(() => { refresh(); }, [refresh, refreshKey]);

  useEffect(() => {
    if (!docs.some((doc) => ACTIVE_STATUSES.has(doc.status))) return undefined;
    const timer = window.setInterval(refresh, 2000);
    return () => window.clearInterval(timer);
  }, [docs, refresh]);

  const openChunks = async (record) => {
    setSelectedDoc(record);
    setDrawerOpen(true);
    setChunks([]);
    setChunksLoading(true);
    try {
      const data = await getDocumentChunks(kbId, record.id);
      setChunks(data.chunks || []);
      if (record.status !== 'ready') {
        message.info('文档尚未索引完成，暂无分块内容');
      }
    } catch (err) {
      message.error(err.message);
      setChunks([]);
    } finally {
      setChunksLoading(false);
    }
  };

  const handleDelete = async (docId) => {
    try {
      await deleteDocument(kbId, docId);
      if (selectedDoc?.id === docId) {
        setDrawerOpen(false);
        setSelectedDoc(null);
        setChunks([]);
      }
      refresh();
    } catch (err) {
      message.error(err.message || '删除失败');
    }
  };

  const columns = [
    {
      title: '文件名',
      dataIndex: 'filename',
      key: 'filename',
      ellipsis: true,
      render: (text, record) => (
        <Button type="link" style={{ padding: 0 }} onClick={(e) => { e.stopPropagation(); openChunks(record); }}>
          {text}
        </Button>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 110,
      render: (status, record) => {
        const cfg = STATUS_MAP[status] || { color: 'default', text: status };
        return (
          <Tag color={cfg.color} title={record.error_message || ''}>
            {cfg.text}
          </Tag>
        );
      },
    },
    { title: '分块数', dataIndex: 'chunk_count', key: 'chunk_count', width: 80, align: 'center' },
    {
      title: '上传时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 160,
      render: (value) => dayjs(value).format('YYYY-MM-DD HH:mm'),
    },
    {
      title: '操作',
      key: 'action',
      width: 120,
      align: 'center',
      render: (_, record) => (
        <Space size={4} onClick={(e) => e.stopPropagation()}>
          <Button type="text" icon={<EyeOutlined />} onClick={() => openChunks(record)} />
          <Popconfirm title="确定删除此文档？" onConfirm={() => handleDelete(record.id)}>
            <Button type="text" danger icon={<DeleteOutlined />} disabled={record.status === 'deleting'} />
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <Typography.Text strong>文档列表</Typography.Text>
        <Button icon={<ReloadOutlined />} size="small" onClick={refresh} loading={loading}>刷新</Button>
      </div>
      <Table
        columns={columns}
        dataSource={docs}
        rowKey="id"
        loading={loading}
        size="middle"
        locale={{ emptyText: '暂无文档，请上传' }}
        pagination={{ pageSize: 20, showSizeChanger: false, showTotal: (total) => `共 ${total} 个文档` }}
        onRow={(record) => ({
          onClick: () => openChunks(record),
          style: { cursor: 'pointer' },
        })}
      />

      <Drawer
        title={selectedDoc?.filename || '文档分块'}
        width={720}
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        destroyOnClose
      >
        {chunksLoading ? (
          <div style={{ textAlign: 'center', padding: 48 }}><Spin /></div>
        ) : chunks.length === 0 ? (
          <Empty description="暂无分块内容" />
        ) : (
          <List
            dataSource={chunks}
            split={false}
            renderItem={(chunk) => (
              <List.Item style={{ display: 'block', padding: '0 0 16px' }}>
                <div style={{ border: '1px solid #f0f0f0', borderRadius: 8, padding: 16, background: '#fff' }}>
                  <Space size={8} style={{ marginBottom: 8 }}>
                    <Tag color="blue">#{chunk.chunk_index}</Tag>
                    {chunk.page ? <Tag>第 {chunk.page} 页</Tag> : null}
                  </Space>
                  <Typography.Paragraph style={{ whiteSpace: 'pre-wrap', marginBottom: 0 }}>
                    {chunk.content}
                  </Typography.Paragraph>
                </div>
              </List.Item>
            )}
          />
        )}
      </Drawer>
    </div>
  );
}
