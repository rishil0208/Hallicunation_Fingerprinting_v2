#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "[*] Starting FastAPI backend on http://127.0.0.1:8000..."
.venv/bin/uvicorn backend.app.api.routes:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

cleanup() {
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $VITE_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[*] Starting Vite frontend on http://127.0.0.1:5173..."
cd "$DIR/frontend"
node -e "
import('vite').then(async ({ createServer }) => {
  const server = await createServer({
    configFile: './vite.config.js',
    server: { host: '127.0.0.1', port: 5173 }
  });
  server.httpServer.listen(5173, '127.0.0.1', () => {
    console.log('Vite running at http://127.0.0.1:5173/');
  });
}).catch(console.error);
" &
VITE_PID=$!

wait
