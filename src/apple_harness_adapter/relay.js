// Harness loopback HTTP, carried over Docker exec's stdin and stdout.
const http = require("node:http");
const readline = require("node:readline");

const pending = new Map();
let nextId = 0;
const server = http.createServer(async (request, response) => {
  if (request.method !== "POST" || request.url !== "/v1/chat/completions") {
    response.writeHead(404).end();
    return;
  }
  let size = 0;
  const chunks = [];
  try {
    for await (const chunk of request) {
      size += chunk.length;
      if (size > 1024 * 1024) {
        response.writeHead(413).end();
        return;
      }
      chunks.push(chunk);
    }
    const id = nextId++;
    pending.set(id, response);
    response.on("close", () => pending.delete(id));
    process.stdout.write(JSON.stringify({
      id,
      authorization: request.headers.authorization,
      body: Buffer.concat(chunks).toString("utf8"),
    }) + "\n");
  } catch {
    response.writeHead(400).end();
  }
});

const input = readline.createInterface({ input: process.stdin });
input.on("line", (line) => {
  const { id, status, contentType, body } = JSON.parse(line);
  const response = pending.get(id);
  if (response) response.writeHead(status, { "Content-Type": contentType }).end(body);
});
input.on("close", () => process.exit(0));
server.listen(18765, "127.0.0.1", () => process.stdout.write("ready\n"));
