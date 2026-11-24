FROM node:24-bookworm-slim@sha256:873f586f3794f52281bd039bf02c048a0be08fde619338f9941bad1eb6b34530

RUN install -d -o node -g node /workspace /home/node/.nvm \
    && touch /home/node/.nvm/nvm.sh
COPY package.json package-lock.json /opt/harbor-pi/
RUN npm ci --prefix /opt/harbor-pi --omit=dev --ignore-scripts \
    && ln -s /opt/harbor-pi/node_modules/.bin/pi /usr/local/bin/pi \
    && test "$(pi --version)" = 1.5.2
WORKDIR /workspace
