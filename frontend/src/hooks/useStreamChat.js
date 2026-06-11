import { useState, useRef, useCallback } from 'react';
import { streamChat } from '../api/chat';

export function useStreamChat(sessionId) {
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef(null);

  const startStream = useCallback(
    (question, params, { onToken, onDone, onError, onSources, onStatus }) => {
      const controller = new AbortController();
      abortRef.current = controller;
      setStreaming(true);

      streamChat(
        sessionId,
        { question, ...params },
        {
          onToken: (token) => {
            onToken?.(token);
          },
          onSources: (sources) => {
            onSources?.(sources);
          },
          onStatus: (status) => {
            onStatus?.(status);
          },
          onDone: () => {
            setStreaming(false);
            abortRef.current = null;
            onDone?.();
          },
          onError: (msg) => {
            setStreaming(false);
            abortRef.current = null;
            onError?.(msg);
          },
          signal: controller.signal,
        }
      );
    },
    [sessionId]
  );

  const stopStream = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
      setStreaming(false);
    }
  }, []);

  return { streaming, startStream, stopStream };
}
