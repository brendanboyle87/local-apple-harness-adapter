FROM node:24-bookworm-slim@sha256:873f586f3794f52281bd039bf02c048a0be08fde619338f9941bad1eb6b34530

RUN install -d -o node -g node /workspace
COPY package.json package-lock.json /opt/harbor-opencode/
RUN npm ci --prefix /opt/harbor-opencode --omit=dev --ignore-scripts \
    && ln -s /opt/harbor-opencode/node_modules/opencode-linux-arm64/bin/opencode /usr/local/bin/opencode \
    && test "$(opencode --version)" = 1.18.30
WORKDIR /workspace
