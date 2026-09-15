#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
  exec /opt/ma-sub/mas "$@"
fi

[[ -n "${PUBLIC_KEY:-}" ]] || { echo "PUBLIC_KEY is required for the controller" >&2; exit 1; }
umask 077
install -d -m 700 /root/.ssh
printf '%s\n' "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A
install -d -m 755 /run/sshd
exec /usr/sbin/sshd -D -e -o PasswordAuthentication=no \
  -o PermitRootLogin=prohibit-password -o AuthenticationMethods=publickey
