import { useState } from 'react';
import { Upload, message as antMsg, Spin } from 'antd';
import { InboxOutlined } from '@ant-design/icons';
import { uploadDocument } from '../api/documents';

const { Dragger } = Upload;

export default function UploadZone({ kbId, onUploaded }) {
  const [uploading, setUploading] = useState(false);
  const [progressFile, setProgressFile] = useState(null);

  const handleUpload = async (file) => {
    setUploading(true);
    setProgressFile(file.name);
    antMsg.loading({ content: `正在解析 ${file.name}...`, key: 'upload', duration: 0 });
    try {
      const result = await uploadDocument(kbId, file);
      antMsg.success({ content: `${file.name} 上传成功`, key: 'upload' });
      onUploaded?.();
      return result;
    } catch (err) {
      antMsg.error({ content: `${file.name} 上传失败: ${err.message}`, key: 'upload' });
      throw err;
    } finally {
      setUploading(false);
      setProgressFile(null);
    }
  };

  return (
    <div className="upload-dragger" style={{ marginBottom: 16 }}>
      <Dragger
        name="file"
        multiple
        showUploadList={false}
        customRequest={({ file, onSuccess, onError }) => {
          handleUpload(file).then(onSuccess).catch(onError);
        }}
        accept=".pdf,.doc,.docx,.xls,.xlsx,.ppt,.pptx,.txt,.md,.csv,.png,.jpg,.jpeg"
        disabled={uploading}
      >
        {uploading ? (
          <>
            <Spin size="large" />
            <p className="ant-upload-text" style={{ marginTop: 12 }}>正在解析中...</p>
            <p className="ant-upload-hint">{progressFile}</p>
          </>
        ) : (
          <>
            <p className="ant-upload-drag-icon"><InboxOutlined /></p>
            <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
            <p className="ant-upload-hint">
              支持 PDF / Word / Excel / PPT / TXT / Markdown / CSV / 图片
            </p>
          </>
        )}
      </Dragger>
    </div>
  );
}
