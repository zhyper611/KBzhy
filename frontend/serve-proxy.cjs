const http = require('http')
const fs = require('fs')
const path = require('path')

const PORT = 3000
const DIST = path.join(__dirname, 'dist')

const MIME = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
}

function serveFile(req, res, filePath) {
  const ext = path.extname(filePath)
  res.writeHead(200, { 'Content-Type': MIME[ext] || 'application/octet-stream' })
  fs.createReadStream(filePath).pipe(res)
}

function proxyApi(req, res) {
  const opts = {
    hostname: '127.0.0.1',
    port: 8000,
    path: req.url,
    method: req.method,
    headers: req.headers,
  }
  const proxy = http.request(opts, (proxyRes) => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers)
    proxyRes.pipe(res)
  })
  proxy.on('error', () => {
    res.writeHead(502)
    res.end('Bad Gateway')
  })
  req.pipe(proxy)
}

http.createServer((req, res) => {
  if (req.url.startsWith('/api/')) return proxyApi(req, res)

  let filePath = path.join(DIST, req.url === '/' ? 'index.html' : req.url)

  // 安全检查，防止目录穿越
  if (!filePath.startsWith(DIST)) {
    res.writeHead(403)
    return res.end('Forbidden')
  }

  fs.stat(filePath, (err, stats) => {
    if (err || !stats.isFile()) {
      filePath = path.join(DIST, 'index.html')
    }
    serveFile(req, res, filePath)
  })
}).listen(PORT, () => {
  console.log(`http://localhost:${PORT}  (static + /api → 8000)`)
})
