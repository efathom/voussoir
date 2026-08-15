#!/usr/bin/env bash
# Install local-CI git hooks.
# Pre-commit hooks live in `.pre-commit-config.yaml` and are installed by `pre-commit install`.
# This script installs the additional pre-push hook that runs the full `make ci` suite
# (lint + typecheck + tests) before any local `git push`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOOK="$REPO_ROOT/.git/hooks/pre-push"

cat > "$HOOK" <<'EOF'
#!/usr/bin/env bash
# Auto-installed by scripts/install_hooks.sh.
# Runs the full local-CI suite before any push.
set -euo pipefail

echo "→ Running local CI before push…"
exec make -C "$(git rev-parse --show-toplevel)" ci
EOF

chmod +x "$HOOK"
echo "✓ Installed pre-push hook at $HOOK"
