import { useEffect, useState, useCallback } from 'react';
import { Select, Button, Modal, Input, Popconfirm, Space, message } from 'antd';
import { PlusOutlined, DeleteOutlined, DatabaseOutlined } from '@ant-design/icons';
import { listKnowledgeBases, createKnowledgeBase, deleteKnowledgeBase } from '../api/documents';

export default function KBSelector({ value, onChange, disabled, allowCreate = true }) {
  const [kbs, setKbs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [newName, setNewName] = useState('');
  const [creating, setCreating] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setLoading(true);
      const data = await listKnowledgeBases();
      setKbs(data || []);
    } catch {
      setKbs([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const handleCreate = async () => {
    if (!newName.trim()) return;
    setCreating(true);
    try {
      const kb = await createKnowledgeBase(newName.trim());
      message.success(`知识库「${newName.trim()}」已创建`);
      setNewName('');
      setModalOpen(false);
      await refresh();
      onChange?.(kb.kb_id);
    } catch (err) {
      message.error(err.message);
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async () => {
    if (!value) return;
    setDeleting(true);
    try {
      await deleteKnowledgeBase(value);
      const deleted = kbs.find((kb) => kb.kb_id === value);
      message.success(`知识库「${deleted?.name || value}」已删除`);
      onChange?.(null);
      await refresh();
    } catch (err) {
      message.error(err.message);
    } finally {
      setDeleting(false);
    }
  };

  return (
    <>
      <Space.Compact style={{ width: '100%' }}>
        <Select
          value={value}
          onChange={onChange}
          loading={loading}
          placeholder="请选择知识库"
          style={{ flex: 1 }}
          disabled={disabled}
          allowClear
          options={kbs.map((kb) => ({
            value: kb.kb_id,
            label: (
              <span>
                <DatabaseOutlined style={{ marginRight: 6 }} />
                {kb.name}
                <span style={{ color: '#999', marginLeft: 6, fontSize: 12 }}>
                  {kb.doc_count} 文档
                </span>
              </span>
            ),
          }))}
          notFoundContent="暂无知识库，请新建"
        />
        {allowCreate && <Button icon={<PlusOutlined />} onClick={() => setModalOpen(true)} disabled={disabled} />}
        {allowCreate && value && (
          <Popconfirm
            title="确定删除此知识库？"
            description="将同时删除该知识库下的所有文档和向量数据，不可恢复。"
            onConfirm={handleDelete}
            okText="确定删除"
            cancelText="取消"
            okButtonProps={{ danger: true, loading: deleting }}
          >
            <Button icon={<DeleteOutlined />} danger disabled={disabled || deleting} />
          </Popconfirm>
        )}
      </Space.Compact>

      {allowCreate && <Modal
        title="新建知识库"
        open={modalOpen}
        onOk={handleCreate}
        onCancel={() => { setModalOpen(false); setNewName(''); }}
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
      </Modal>}
    </>
  );
}
