import { lazy, Suspense, useMemo } from 'react';
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import {
  Layout, Menu, ConfigProvider, Typography, Spin,
} from 'antd';
import { MessageOutlined, FileTextOutlined } from '@ant-design/icons';
import zhCN from 'antd/locale/zh_CN';

const ChatPage = lazy(() => import('./pages/ChatPage'));
const DocumentsPage = lazy(() => import('./pages/DocumentsPage'));

const { Sider } = Layout;

export default function App() {
  const navigate = useNavigate();
  const location = useLocation();

  const current = location.pathname === '/documents' ? 'documents' : 'chat';

  const themeConfig = useMemo(() => ({
    token: { colorPrimary: '#1677ff', borderRadius: 6 },
  }), []);

  return (
    <ConfigProvider theme={themeConfig} locale={zhCN}>
      <Layout style={{ height: '100vh' }}>
        <Sider
          breakpoint="lg"
          collapsedWidth="0"
          theme="light"
          style={{ background: '#fff', borderRight: '1px solid #f0f0f0' }}
        >
          <div style={{ padding: '16px', textAlign: 'center' }}>
            <Typography.Title level={4} style={{ margin: 0, color: '#1677ff' }}>
              KBzhy
            </Typography.Title>
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              知识库问答系统
            </Typography.Text>
          </div>
          <Menu
            mode="inline"
            selectedKeys={[current]}
            onClick={({ key }) => navigate(key === 'documents' ? '/documents' : '/')}
            items={[
              { key: 'chat', icon: <MessageOutlined />, label: '对话问答' },
              { key: 'documents', icon: <FileTextOutlined />, label: '文档管理' },
            ]}
            style={{ borderInlineEnd: 'none' }}
          />
        </Sider>
        <Layout>
          <Suspense fallback={(
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
              <Spin />
            </div>
          )}
          >
            <Routes>
              <Route path="/" element={<ChatPage />} />
              <Route path="/documents" element={<DocumentsPage />} />
            </Routes>
          </Suspense>
        </Layout>
      </Layout>
    </ConfigProvider>
  );
}
