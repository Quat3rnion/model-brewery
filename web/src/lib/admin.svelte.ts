// Admin key for this tab only. Kept in memory: not in storage, the URL or logs.
// A reload or "Sign out" forgets it; a 401 from the server also clears it.

class AdminSession {
  key = $state<string | null>(null);
  set(k: string) {
    this.key = k;
  }
  clear() {
    this.key = null;
  }
}

export const adminSession = new AdminSession();
