import { useState } from 'react';
import { Card, Tag, Typography, Alert, Modal } from 'antd';
import {
  UserOutlined, RobotOutlined, WarningFilled, LinkOutlined, DownOutlined,
  SearchOutlined, ReloadOutlined, ThunderboltOutlined,
} from '@ant-design/icons';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';

const STAGE_ICONS = {
  rewrite: <ReloadOutlined />,
  expand: <SearchOutlined />,
  retrieve: <SearchOutlined />,
  rerank: <ThunderboltOutlined />,
  generate: <ThunderboltOutlined />,
};

export default function ChatBubble({ message }) {
  const [expanded, setExpanded] = useState(false);
  const [selectedSource, setSelectedSource] = useState(null);
  const isUser = message.role === 'user';

  const sourceCount = message.sources?.length || 0;
  const hasHallucination = message.hallucination_score != null && message.hallucination_score > 0.5;
  const showStatus = message.status && message.isStreaming;

  return (
    <div style={{
      display: 'flex', justifyContent: isUser ? 'flex-end' : 'flex-start',
      marginBottom: 16, paddingLeft: isUser ? 48 : 0, paddingRight: isUser ? 0 : 48,
    }}>
      <div style={{ maxWidth: '85%' }}>
        {/* Role indicator */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4,
          flexDirection: isUser ? 'row-reverse' : 'row',
        }}>
          {isUser ? <UserOutlined /> : <RobotOutlined />}
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            {isUser ? '你' : 'KBzhy'}
          </Typography.Text>
          {hasHallucination && (
            <Tag color="warning" icon={<WarningFilled />}>可能不准确</Tag>
          )}
        </div>

        {/* Content bubble */}
        <Card
          size="small"
          styles={{
            body: {
              padding: '10px 14px',
              background: isUser ? '#e6f4ff' : undefined,
              borderRadius: 8,
            },
          }}
        >
          {isUser ? (
            <Typography.Text>{message.content}</Typography.Text>
          ) : message.isStreaming && !message.content ? (
            <div className="rag-status-display">
              {showStatus ? (
                <>
                  <span className="rag-status-icon">{STAGE_ICONS[message.status.stage] || <SearchOutlined />}</span>
                  <Typography.Text type="secondary">{message.status.message}</Typography.Text>
                  <span className="thinking-dots"><span className="dot" /><span className="dot" /><span className="dot" /></span>
                </>
              ) : (
                <Typography.Text type="secondary">正在思考<span className="thinking-dots"><span className="dot" /><span className="dot" /><span className="dot" /></span></Typography.Text>
              )}
            </div>
          ) : (
            <div className="markdown-body">
              {showStatus && (
                <div className="rag-status-inline">
                  <span className="rag-status-icon">{STAGE_ICONS[message.status.stage] || <SearchOutlined />}</span>
                  <Typography.Text type="secondary" style={{ fontSize: 12 }}>{message.status.message}</Typography.Text>
                  <span className="thinking-dots"><span className="dot" /><span className="dot" /><span className="dot" /></span>
                </div>
              )}
              <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>{message.content}</ReactMarkdown>
            </div>
          )}
        </Card>

        {/* Sources */}
        {!isUser && sourceCount > 0 && (
          <div style={{ marginTop: 4 }}>
            <Typography.Link
              onClick={() => setExpanded(!expanded)}
              style={{ fontSize: 12 }}
            >
              <LinkOutlined /> {sourceCount} 个引用来源 <DownOutlined rotate={expanded ? 180 : 0} />
            </Typography.Link>
            {expanded && (
              <div style={{ marginTop: 4 }}>
                {message.sources.map((s, i) => (
                  <div
                    key={i}
                    onClick={() => setSelectedSource(s)}
                    style={{
                      padding: '4px 8px', marginBottom: 2,
                      background: 'rgba(0,0,0,0.02)', borderRadius: 4, fontSize: 12,
                      cursor: 'pointer',
                      transition: 'background 0.2s',
                    }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(22,119,255,0.08)'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(0,0,0,0.02)'; }}
                  >
                    <Typography.Text strong>{s.filename || s.source}</Typography.Text>
                    {s.page && <Typography.Text type="secondary" style={{ marginLeft: 4 }}>第{s.page}页</Typography.Text>}
                    <Typography.Paragraph
                      style={{ margin: 0, fontSize: 12 }}
                      type="secondary"
                      ellipsis={{ rows: 2 }}
                    >
                      {s.content || s.text || s.snippet || ''}
                    </Typography.Paragraph>
                  </div>
                ))}
              </div>
            )}
            <Modal
              title={selectedSource ? (selectedSource.filename || selectedSource.source) : '来源详情'}
              open={!!selectedSource}
              onCancel={() => setSelectedSource(null)}
              footer={null}
              width={680}
            >
              {selectedSource && (
                <div style={{ maxHeight: 400, overflow: 'auto', whiteSpace: 'pre-wrap', fontSize: 14, lineHeight: 1.8 }}>
                  {selectedSource.content || selectedSource.text || selectedSource.snippet}
                </div>
              )}
            </Modal>
          </div>
        )}

        {/* Hallucination warning detail */}
        {hasHallucination && (
          <Alert
            type="warning"
            message={`可信度评分: ${((1 - message.hallucination_score) * 100).toFixed(0)}%`}
            style={{ marginTop: 4, fontSize: 12 }}
            size="small"
            showIcon
          />
        )}
      </div>
    </div>
  );
}
