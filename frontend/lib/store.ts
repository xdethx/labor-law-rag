// Minimal Web-Storage-backed external store for useSyncExternalStore.
// The parsed value is cached against the raw string so getSnapshot returns a
// referentially stable value; the server snapshot is always null (storage
// only exists in the browser — this is what avoids hydration mismatches).

export interface StorageStore<T> {
  subscribe(listener: () => void): () => void;
  getSnapshot(): T | null;
  getServerSnapshot(): T | null;
  set(value: T): void;
  update(updater: (prev: T | null) => T): void;
  clear(): void;
}

export function createStorageStore<T>(
  key: string,
  getStorage: () => Storage,
): StorageStore<T> {
  const listeners = new Set<() => void>();
  let cachedRaw: string | null = null;
  let cachedValue: T | null = null;

  function emit(): void {
    listeners.forEach((listener) => listener());
  }

  function getSnapshot(): T | null {
    let raw: string | null = null;
    try {
      raw = getStorage().getItem(key);
    } catch {
      return null;
    }
    if (raw !== cachedRaw) {
      cachedRaw = raw;
      try {
        cachedValue = raw ? (JSON.parse(raw) as T) : null;
      } catch {
        cachedValue = null;
      }
    }
    return cachedValue;
  }

  function set(value: T): void {
    getStorage().setItem(key, JSON.stringify(value));
    emit();
  }

  return {
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    getSnapshot,
    getServerSnapshot: () => null,
    set,
    update(updater) {
      set(updater(getSnapshot()));
    },
    clear() {
      getStorage().removeItem(key);
      emit();
    },
  };
}
