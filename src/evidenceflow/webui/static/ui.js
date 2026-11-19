/* 工作台共用的 DOM、Markdown、来源页码和 SSE 读取逻辑。 */
import MarkdownIt from "./markdown-it.min.js";

export const byId = (id) => document.getElementById(id);

export function element(tag, className = "", text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export function icon(name) {
  const node = element("span", "icon i-" + name);
  node.setAttribute("aria-hidden", "true");
  return node;
}

export function action(label, image, className, activate) {
  const button = element("button", className);
  button.type = "button";
  if (image) button.append(icon(image));
  button.append(document.createTextNode(label));
  button.addEventListener("click", activate);
  return button;
}

export function emptyState(title, description, shortcut) {
  const node = element("div", "empty-state");
  const art = element("div", "empty-art");
  const file = element("span", "file-icon");
  file.append(icon("file"));
  art.append(file);
  node.append(art, element("h3", "", title), element("p", "", description));
  if (shortcut) node.append(shortcut);
  return node;
}

export function pageRange(source) {
  const { page_start: start, page_end: end } = source;
  if (start == null) return "无页码";
  return start === end ? "第 " + start + " 页" : "第 " + start + "–" + end + " 页";
}

export function passageCard(passage, title = "", highlight = false) {
  // 片段是索引文本，保留断行，标题和正文均作为纯文本插入。
  const node = element("article", highlight ? "passage-card anchor" : "passage-card");
  const location = title ? title + " · " + pageRange(passage) : pageRange(passage);
  node.append(element("div", "passage-meta", location));
  if (passage.heading) node.append(element("div", "passage-section", passage.heading));
  node.append(element("div", "passage-body", passage.content));
  return node;
}

export const formatCount = new Intl.NumberFormat("zh-CN");

export function duration(ms) {
  if (ms == null) return "";
  return ms >= 1000 ? (ms / 1000).toFixed(1) + " 秒" : Math.round(ms) + " ms";
}

const markdown = new MarkdownIt({ html: false, breaks: true }).disable("image");

export function renderMarkdown(text, sources = [], onCitation) {
  const node = element("div", "markdown");
  // 模型和浏览器历史中的文本都按不允许 HTML 的 Markdown 重新渲染。
  node.innerHTML = markdown.render(text);
  node.querySelectorAll("a").forEach((link) => {
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  });
  if (!sources.length) return node;

  const labels = new Map(sources.map((source) => [source.source_label, source]));
  const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
  const replacements = [];
  // 只替换已校验的标签，保留代码示例和链接原有语义。
  for (let leaf = walker.nextNode(); leaf; leaf = walker.nextNode()) {
    if (leaf.parentElement.closest("a, code, pre")) continue;
    const hits = [...leaf.data.matchAll(/\[S[0-9]+\]/g)].filter(([label]) => labels.has(label));
    if (hits.length) replacements.push([leaf, hits]);
  }
  for (const [leaf, hits] of replacements) {
    const content = document.createDocumentFragment();
    let start = 0;
    for (const { 0: label, index } of hits) {
      const source = labels.get(label);
      const button = action(label.slice(1, -1), null, "citation-chip", () => onCitation(source));
      button.setAttribute("aria-label", "查看引文 " + label + " · " + source.file_name);
      button.title = source.file_name + " · " + pageRange(source);
      content.append(leaf.data.slice(start, index), button);
      start = index + label.length;
    }
    content.append(leaf.data.slice(start));
    leaf.replaceWith(content);
  }
  return node;
}

export function postJSON(body) {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

export async function request(path, options) {
  let response;
  try {
    response = await fetch(path, { cache: "no-store", ...options });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new Error("无法连接本地工作台，请检查服务是否正在运行。");
  }
  if (response.ok) return response;
  const problem = await response.json().catch(() => ({}));
  // 开发环境里静态文件可能先于后端进程更新。保留文档范围，提示重启旧后端。
  const fields = Array.isArray(problem?.detail) ? problem.detail : [];
  if (path === "/api/ask" && response.status === 422 && fields.some((field) =>
    field?.type === "extra_forbidden" &&
      ["document_id", "thread_id", "turn_id", "settings", "force_search"].includes(field.loc?.at(-1))
  )) {
    throw new Error("当前 WebUI 后端版本较旧（HTTP 422）。请重启 WebUI 服务后重新提问。");
  }
  const message = typeof problem?.detail === "string" ? problem.detail : "请求未完成，请重试。";
  throw new Error(`HTTP ${response.status} · ${message}`);
}

export async function json(path, options) {
  return (await request(path, options)).json();
}

export async function askStream(payload, onFrame) {
  // 一个网络分片可能包含半帧或多帧，UTF-8 解码和 SSE 分帧各自跨分片保留状态。
  const response = await request("/api/ask", postJSON(payload));
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;
      for (let end = buffer.indexOf("\n\n"); end >= 0; end = buffer.indexOf("\n\n")) {
        const frame = buffer.slice(0, end);
        buffer = buffer.slice(end + 2);
        if (frame.startsWith(":")) continue;
        const [event, data] = frame.split("\n");
        const type = event.slice("event: ".length);
        const payload = JSON.parse(data.slice("data: ".length));
        if (type === "result") return payload;
        onFrame(type, payload);
      }
    }
  } finally {
    reader.releaseLock();
  }
  throw new Error("问答连接已中断，可重新提问。");
}
