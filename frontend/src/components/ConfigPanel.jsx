import { Form, Slider, Select, InputNumber, Collapse, Switch } from 'antd';
import { SettingOutlined } from '@ant-design/icons';

const CHAIN_OPTIONS = [
  { value: 'stuff', label: 'Stuff — 单次摘要' },
  { value: 'map_reduce', label: 'Map-Reduce — 并行摘要合并' },
  { value: 'refine', label: 'Refine — 逐步迭代精炼' },
];

const RERANK_OPTIONS = [
  { value: 'model', label: '模型重排 — 专用 Reranker 打分' },
  { value: 'llm', label: 'LLM 重排 — 大模型判断相关性' },
  { value: 'keyword', label: '关键词重排 — 纯文本匹配' },
];

export default function ConfigPanel({ values, onChange }) {
  const [form] = Form.useForm();

  const handleChange = (_, all) => {
    onChange?.(all);
  };

  return (
    <Collapse
      ghost
      defaultActiveKey={['config']}
      items={[{
        key: 'config',
        label: <span><SettingOutlined /> 参数配置</span>,
        children: (
          <Form
            form={form}
            layout="vertical"
            size="small"
            initialValues={{ top_k: 5, temperature: 0.5, chain_type: 'stuff', rerank_method: 'model', similarity_threshold: 0.35, enable_expansion: false, ...values }}
            onValuesChange={handleChange}
          >
            <Form.Item label="top_k" name="top_k">
              <InputNumber min={1} max={20} style={{ width: '100%' }} />
            </Form.Item>
            <Form.Item label="温度 (temperature)" name="temperature">
              <Slider min={0} max={1} step={0.1} marks={{ 0: '精确', 0.5: '平衡', 1: '创意' }} />
            </Form.Item>
            <Form.Item label="检索链类型" name="chain_type">
              <Select options={CHAIN_OPTIONS} />
            </Form.Item>
            <Form.Item label="重排序方法" name="rerank_method">
              <Select options={RERANK_OPTIONS} />
            </Form.Item>
            <Form.Item label="相似度阈值" name="similarity_threshold">
              <Slider min={0} max={1} step={0.05} marks={{ 0: '全收', 0.35: '默认', 0.7: '严格' }} />
            </Form.Item>
            <Form.Item label="查询扩展" name="enable_expansion" valuePropName="checked">
              <Switch />
            </Form.Item>
          </Form>
        ),
      }]}
    />
  );
}
