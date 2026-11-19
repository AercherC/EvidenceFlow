/* 聊天工作台的会话状态、请求生命周期和证据联动。 */
import { initLibrary } from "./library.js";
import { BUDGET_LABELS, budgetValue, initAgentSettings } from "./settings.js";
import {
  action, askStream, byId, duration, element, emptyState, formatCount, icon,
  json, pageRange, passageCard, postJSON, renderMarkdown, request,
} from "./ui.js";

const THREAD_KEY = "evidenceflow.thread.v1.";
const ACTIVE_KEY = "evidenceflow.active-thread.v1";
const workspace = byId("workspace");
const question = byId("question");
const chatScroll = byId("chat-scroll");
const evidenceContent = byId("evidence-content");
const medium = matchMedia("(max-width: 1100px)");
const phone = matchMedia("(max-width: 740px)");
const threadDate = new Intl.DateTimeFormat("zh-CN", { month: "short", day: "numeric" });

let threads = readHistory();
let thread = threads.find((item) => item.id === lastOpened()) ?? threads[0] ?? newThread();
let view = "chat";
let evidenceTab = "citations";
let evidenceOpen = true;
let drawer = null;
let running = null;
let citation = null;
let dialogAction = null;
let toastTimer;

function newThread() {
  return { id: crypto.randomUUID(), title: "", turns: [], updated: Date.now(), evidence: null, draft: "" };
}

function interrupted(steps, error) {
  return {
    status: "failed", answer: "", sources: [], error,
    steps: steps.map((step) => step.phase === "started" ? { ...step, phase: "interrupted", detail: "中断" } : step),
  };
}

function readHistory() {
  const saved = [];
  try {
    for (let index = 0; index < localStorage.length; index += 1) {
      const key = localStorage.key(index);
      if (!key.startsWith(THREAD_KEY)) continue;
      let data;
      try {
        data = JSON.parse(localStorage.getItem(key));
      } catch {
        continue;
      }
      if (data?.id !== key.slice(THREAD_KEY.length) || !Array.isArray(data.turns)) continue;
      // 刷新时，未收到终态的旧 SSE 不再显示为运行中。
      for (const turn of data.turns) {
        turn.steps = turn.result?.steps ?? [];
        turn.result ??= interrupted(turn.steps, "页面已重新打开，本轮连接已中断。");
        turn.evidenceActions = turn.evidenceActions?.map((row) =>
          row.phase === "started" ? { ...row, phase: "interrupted" } : row);
      }
      saved.push({ ...data, draft: "" });
    }
  } catch {
    // 禁用浏览器存储时仍可使用当前页面。
  }
  return saved.sort((a, b) => b.updated - a.updated);
}

function lastOpened() {
  try {
    return sessionStorage.getItem(ACTIVE_KEY);
  } catch {
    return null;
  }
}

function persist(item) {
  // 每条会话独立存储，刷新可恢复；原文总是按索引版本从服务端重新读取。
  try {
    const { draft, draftScope, ...data } = item;
    localStorage.setItem(THREAD_KEY + item.id, JSON.stringify(data));
  } catch {
    toast("浏览器无法保存对话，最近更改仅保留在当前页。");
  }
}

function titleOf(item) {
  return item.title || item.turns[0]?.query.replace(/\s+/g, " ").slice(0, 70) || "新对话";
}

function toast(message) {
  clearTimeout(toastTimer);
  byId("toast").textContent = message;
  byId("toast").hidden = false;
  toastTimer = setTimeout(() => { byId("toast").hidden = true; }, 3400);
}

function showHistory() {
  const term = byId("thread-search").value.trim().toLowerCase();
  const history = byId("history-list");
  const matching = threads.filter((item) => {
    return [titleOf(item), ...item.turns.map((turn) => turn.query)].some((text) => text.toLowerCase().includes(term));
  });
  history.replaceChildren();
  if (!matching.length) {
    history.append(element("li", "list-empty", term ? "未找到匹配的对话。" : "暂无已保存的对话。"));
  }
  for (const item of matching) {
    const label = element("span");
    label.append(element("strong", "", titleOf(item)), element("small", "", threadDate.format(item.updated)));
    const button = action("", "messages", "thread-item", () => openThread(item));
    button.append(label);
    button.title = titleOf(item);
    if (item === thread) button.setAttribute("aria-current", "true");
    const row = element("li");
    row.append(button);
    history.append(row);
  }
}

function visibleEvidence() {
  return view === "chat" && (medium.matches ? drawer === "evidence" : evidenceOpen);
}

function chrome() {
  const hasDrawer = drawer !== null;
  workspace.dataset.view = view;
  workspace.dataset.evidence = evidenceOpen ? "open" : "closed";
  workspace.dataset.drawer = drawer ?? "";
  byId("drawer-backdrop").hidden = !hasDrawer;
  byId("open-history").setAttribute("aria-expanded", String(drawer === "history"));
  byId("toggle-evidence").hidden = view !== "chat";
  byId("toggle-evidence").setAttribute("aria-expanded", String(visibleEvidence()));
  byId("chat-screen").hidden = view !== "chat";
  byId("library-screen").hidden = view !== "library";
  byId("settings-screen").hidden = view !== "settings";
  for (const name of ["chat", "library", "settings"]) {
    const button = byId("nav-" + name);
    if (name === view) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  }
  document.querySelector(".main-pane").inert = hasDrawer;
  byId("sidebar").inert = drawer === "evidence" || (phone.matches && drawer !== "history");
  byId("evidence").inert = !visibleEvidence();
}

function showView(next) {
  view = next;
  drawer = null;
  chrome();
  if (view === "chat") showEvidence();
  else if (view === "library") library.checkImports();
  else agentSettings.load();
}

function closeDrawer() {
  if (!drawer) return;
  const toggle = drawer === "history" ? "open-history" : "toggle-evidence";
  drawer = null;
  chrome();
  byId(toggle).focus();
}

function revealEvidence() {
  const openingDrawer = medium.matches && drawer !== "evidence";
  if (medium.matches) drawer = "evidence";
  else evidenceOpen = true;
  chrome();
  if (openingDrawer) byId("close-evidence").focus();
}

function openThread(item) {
  thread.draft = question.value;
  thread = item;
  question.value = thread.draft;
  byId("force-search").checked = false;
  try {
    sessionStorage.setItem(ACTIVE_KEY, thread.id);
  } catch {
    // 当前页仍可切换会话。
  }
  evidenceTab = running?.thread === thread ? "trace" : "citations";
  showView("chat");
  showHistory();
  showChat({ toEnd: true });
}

function currentEvidence() {
  return thread.turns.find((turn) => turn.id === thread.evidence?.turnId) ?? thread.turns.at(-1);
}

function selectEvidence(turn, tab, source) {
  const prior = thread.evidence?.turnId === turn.id ? thread.evidence : null;
  thread.evidence = {
    turnId: turn.id,
    label: source?.source_label ?? prior?.label ?? turn.result?.sources[0]?.source_label,
    context: source?.source_label === prior?.label || !source ? Boolean(prior?.context) : false,
  };
  evidenceTab = tab;
  persist(thread);
  revealEvidence();
  evidenceContent.scrollTop = 0;
  showEvidence();
  if (source) byId("citation-reader")?.scrollIntoView({ block: "nearest" });
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast("已复制到剪贴板。");
  } catch {
    toast("无法访问剪贴板，可直接选中文字复制。");
  }
}

function runSummary(turn) {
  if (!turn.result) return "正在取证";
  const { searches, sources } = turn.result;
  return (searches ?? 0) + " 轮 · " + sources.length + " 条引文";
}

function renderTurn(turn) {
  const result = turn.result;
  const message = element("article", "turn");
  message.id = "turn-" + turn.id;
  const prompt = element("div", "question-message");
  prompt.append(element("p", "", turn.query));
  const reply = element("div", "assistant-message");
  const avatar = element("span", "brand-mark assistant-mark");
  avatar.append(icon("book"));
  const body = element("div", "message-body");
  const label = element("header", "assistant-label");
  label.append(element("strong", "", "EvidenceFlow"));

  const statusLabels = { insufficient_context: "证据不足", failed: "处理失败", cancelled: "已停止" };
  const statusText = result ? statusLabels[result.status] : "检索中";
  if (statusText) label.append(element("span", result ? "pill warning" : "pill", statusText));
  body.append(label);

  if (!result || result.trace || result.settings) {
    const traceButton = action("取证记录", "route", "run-summary", () => selectEvidence(turn, "trace"));
    traceButton.append(element("small", "", runSummary(turn)), icon("chevron-right"));
    if (!result) traceButton.querySelector("small").setAttribute("role", "status");
    body.append(traceButton);
  }
  const actions = element("div", "message-actions");
  if (result?.answer) {
    body.append(renderMarkdown(result.answer, result.sources, (source) => selectEvidence(turn, "citations", source)));
    actions.append(action("复制回答", "copy", "quiet-button", () => copyText(result.answer)));
    if (result.sources.length) {
      actions.append(action("查看 " + result.sources.length + " 条引文", "file", "quiet-button", () => selectEvidence(turn, "citations")));
    }
  } else if (result) {
    const message = result.error ?? "本轮已停止。";
    body.append(element("p", "message-error", message));
    const retry = action("重新提问", "refresh", "quiet-button", () => submit(turn.query, turn));
    retry.disabled = running !== null;
    actions.append(retry);
  }
  if (actions.childElementCount) body.append(actions);
  reply.append(avatar, body);
  message.append(prompt, reply);
  return message;
}

function showComposer() {
  const here = running?.thread === thread;
  const scope = thread.draftScope;
  const clearScope = byId("clear-document-scope");
  clearScope.hidden = !scope;
  byId("document-scope-label").textContent = scope ? "仅检索 " + scope.file_name : "";
  if (scope) clearScope.setAttribute("aria-label", "取消文档范围 " + scope.file_name);
  byId("send-question").hidden = here;
  byId("send-question").disabled = running !== null || !question.value.trim();
  byId("stop-question").hidden = !here;
  byId("stop-question").disabled = !here || !running.id || running.stopping;
  byId("stop-label").textContent = here && running.stopping ? "正在停止…" : "停止";
  const status = byId("composer-status");
  status.textContent = running && !here ? "另一对话正在检索" : "";
  status.hidden = !status.textContent;
  byId("force-search").disabled = Boolean(running);
  byId("fresh-search-control").hidden = thread.settings?.reuse_evidence === false;
  byId("delete-thread").disabled = here;
}

function showChat({ toEnd = false } = {}) {
  const scroll = chatScroll.scrollTop;
  const hasTurns = thread.turns.length > 0;
  byId("thread-title").textContent = titleOf(thread);
  byId("thread-menu").hidden = !hasTurns;
  byId("welcome").hidden = hasTurns;
  const messages = byId("messages");
  messages.hidden = !hasTurns;
  messages.replaceChildren(...thread.turns.map(renderTurn));
  chatScroll.scrollTop = toEnd ? chatScroll.scrollHeight : scroll;
  showComposer();
}

function citationState(turn, source, context) {
  const key = JSON.stringify([turn.id, turn.result.snapshot, source.chunk_id, context]);
  if (citation?.key === key) return citation;
  citation?.controller.abort();
  const controller = new AbortController();
  const entry = { key, controller, data: null, error: "" };
  citation = entry;
  const payload = { source, snapshot: turn.result.snapshot, context };
  json("/api/citation", { ...postJSON(payload), signal: controller.signal })
    .then((data) => { entry.data = data; })
    .catch((error) => { entry.error = error.message; })
    .finally(() => { if (citation === entry && !controller.signal.aborted) showEvidence(); });
  return entry;
}

function showCitations(turn) {
  const sources = turn?.result?.sources ?? [];
  if (!sources.length) {
    let title = "还没有引文";
    let description = "点击回答中的引文编号，这里会显示来源文件、页码和原文。";
    if (turn && !turn.result) {
      title = "正在查找证据";
      description = "回答完成后，可在这里核对实际引用。";
    } else if (turn) {
      title = "本轮没有引用";
      description = "可打开检索过程查看本轮执行情况。";
    }
    const button = turn
      ? action("查看取证记录", "route", "quiet-button", () => setEvidenceTab("trace"))
      : action("浏览文档库", "arrow-right", "quiet-button", () => showView("library"));
    evidenceContent.append(emptyState(title, description, button));
    return;
  }

  const selected = thread.evidence;
  const source = sources.find((item) => item.source_label === selected?.label) ?? sources[0];
  const list = element("ul", "source-list");
  for (const item of sources) {
    const label = element("span", "source-label", item.source_label.replace(/[[\]]/g, ""));
    const file = element("span");
    file.append(element("strong", "", item.file_name), element("small", "", pageRange(item)));
    const button = action("", null, "source-item", () => selectEvidence(turn, "citations", item));
    button.setAttribute("aria-pressed", String(item === source));
    button.append(label, file, icon("chevron-right"));
    const row = element("li");
    row.append(button);
    list.append(row);
  }
  const context = Boolean(selected?.context);
  const result = citationState(turn, source, context);
  const reader = element("div", "citation-reader");
  reader.id = "citation-reader";
  const actions = element("div", "passage-actions");
  actions.append(element("h3", "", "引文原文"));
  if (result.data) {
    const toggle = action(context ? "收起邻块" : "查看邻块", null, "quiet-button", () => {
      thread.evidence = { turnId: turn.id, label: source.source_label, context: !context };
      persist(thread);
      showEvidence();
    });
    toggle.setAttribute("aria-pressed", String(context));
    const copy = action("复制", "copy", "quiet-button", () => copyText(result.data.anchor.content));
    actions.append(element("span"), toggle, copy);
  }
  reader.append(actions);
  if (result.data) {
    for (const previous of result.data.previous) reader.append(passageCard(previous, "前一片段"));
    reader.append(passageCard(result.data.anchor, "引用 " + source.source_label, true));
    for (const following of result.data.following) reader.append(passageCard(following, "后一片段"));
    if (context && !result.data.previous.length && !result.data.following.length) {
      reader.append(element("p", "inline-note", "此引文附近没有已索引的相邻片段。"));
    }
  } else {
    const notice = element("p", "inline-note", result.error || "正在定位原文…");
    notice.setAttribute("role", "status");
    reader.append(notice);
    if (result.error) {
      reader.append(action("重试", "refresh", "quiet-button", () => {
        citation = null;
        showEvidence();
      }));
    }
  }
  evidenceContent.append(list, reader);
}

function passageList(passages, result) {
  const list = element("ul", "evidence-passages");
  for (const passage of passages) {
    if (!passage.file_name) continue;
    const item = element("li");
    const cited = result?.sources.find((source) => source.chunk_id === passage.chunk_id);
    const title = passage.file_name + " · " + pageRange(passage) + (cited ? " · 已引用" : "");
    if (!result?.snapshot) item.textContent = title;
    else {
      const details = element("details");
      const body = element("div", "evidence-inline");
      details.append(element("summary", "", title), body);
      details.addEventListener("toggle", async () => {
        if (!details.open || body.dataset.loaded) return;
        body.dataset.loaded = "true";
        body.textContent = "正在读取原文…";
        try {
          const source = { source_label: passage.source_label ?? "[S1]", document_id: passage.document_id,
            chunk_id: passage.chunk_id, file_name: passage.file_name, page_start: passage.page_start, page_end: passage.page_end };
          const data = await json("/api/citation", postJSON({ snapshot: result.snapshot, source }));
          body.replaceChildren(passageCard(data.anchor, title));
        } catch (error) {
          delete body.dataset.loaded;
          body.textContent = error.message;
        }
      });
      item.append(details);
    }
    list.append(item);
  }
  return list;
}

function showTrace(turn) {
  if (!turn) {
    evidenceContent.append(emptyState("还没有取证记录", "提交问题后，可以在这里查看搜索词、范围和原文。"));
    return;
  }
  const { result } = turn;
  const rows = result?.evidence_actions ?? turn.evidenceActions ?? [];
  const header = element("div", "trace-toolbar");
  header.append(element("strong", "", "取证记录"));
  if (result?.elapsed_ms != null) header.append(element("small", "", duration(result.elapsed_ms)));
  evidenceContent.append(header);

  if (!rows.length) {
    evidenceContent.append(element("p", "inline-note",
      result?.trace ? "详细记录见下方诊断。" : result ? "本轮没有取证记录。" : "等待取证…"));
  }
  const markers = { completed: ["check", "完成"], started: ["route", "进行中"], interrupted: ["close", "中断"] };
  const list = element("ol", "timeline");
  for (const step of rows) {
    const row = element("li");
    const mark = element("span", "timeline-mark " + step.phase);
    mark.setAttribute("role", "img");
    mark.setAttribute("aria-label", markers[step.phase][1]);
    mark.append(icon(markers[step.phase][0]));
    const text = element("div");
    const titles = {
      search: step.index > 1 ? "定向补搜" : "首轮检索", open: "阅读原文",
      reuse: "复用会话证据", context: "用于回答的原文",
    };
    text.append(element("strong", "timeline-title", titles[step.kind]));
    if (step.query) text.append(element("p", "evidence-query", step.query));
    if (step.kind === "search") {
      const scope = step.file_name ?? (step.document_id ? "指定文档" : "当前文档库");
      const outcome = step.hit_count == null ? "" : " · 命中 " + step.hit_count + " 块 · 新增 " + step.passages.length + " 块";
      text.append(element("small", "timeline-detail", scope + outcome));
    }
    if (step.anchor) text.append(element("small", "timeline-detail", step.anchor.file_name + " · " + pageRange(step.anchor)));
    if (step.kind === "reuse" && result?.searches === 0) text.append(element("small", "timeline-detail", "本轮检索 0 次"));
    if (step.kind === "context") text.append(element("small", "timeline-detail", step.tokens + " token · 未纳入 " + step.skipped + " 块"));
    if (step.branches?.length) {
      const details = element("details", "evidence-branches");
      details.append(element("summary", "", "实际分支查询"));
      for (const branch of step.branches) {
        const scope = branch.file_name ?? (branch.document_id ? "指定文档" : "全库");
        details.append(element("p", "", scope + " · " + branch.query));
      }
      text.append(details);
    }
    if (step.passages?.length) text.append(passageList(step.passages, result));
    row.append(mark, text, element("span", "timeline-time", duration(step.duration_ms)));
    list.append(row);
  }
  evidenceContent.append(list);
  if (result?.trace || result?.settings) {
    const details = element("details", "trace-details");
    details.append(element("summary", "", "诊断与本轮预算"));
    if (result.settings) {
      const budget = element("dl", "evidence-budget");
      for (const [key, value] of Object.entries(result.settings)) {
        if (BUDGET_LABELS[key]) budget.append(element("dt", "", BUDGET_LABELS[key]), element("dd", "", budgetValue(value)));
      }
      details.append(budget);
    }
    if (result.trace) details.append(renderMarkdown(result.trace));
    evidenceContent.append(details);
  }
}

function showEvidence() {
  if (!visibleEvidence()) return;
  const turn = currentEvidence();
  byId("citation-count").textContent = turn?.result?.sources.length ?? 0;
  byId("tab-citations").setAttribute("aria-pressed", String(evidenceTab === "citations"));
  byId("tab-trace").setAttribute("aria-pressed", String(evidenceTab === "trace"));
  evidenceContent.replaceChildren();
  if (turn) {
    const scope = element("div", "evidence-question");
    scope.append(element("strong", "", turn.query));
    scope.title = turn.query;
    evidenceContent.append(scope);
  }
  if (evidenceTab === "citations") showCitations(turn);
  else showTrace(turn);
}

function setEvidenceTab(tab) {
  evidenceTab = tab;
  evidenceContent.scrollTop = 0;
  showEvidence();
}

async function submit(text, retry) {
  const query = text.trim();
  if (!query || running) return;
  // 在发送时固定文档身份，重试同一轮时沿用该范围。
  const turn = retry ?? { id: crypto.randomUUID(), query, documentId: thread.draftScope?.document_id };
  turn.steps = [];
  turn.evidenceActions = [];
  turn.result = null;
  if (!retry) {
    thread.turns.push(turn);
    question.value = "";
    thread.draft = "";
    thread.draftScope = null;
  }
  thread.updated = Date.now();
  threads = [thread, ...threads.filter((item) => item !== thread)];
  thread.evidence = { turnId: turn.id };
  evidenceTab = "trace";
  const run = { turn, thread, id: null, stopping: false };
  running = run;
  persist(thread);
  showHistory();
  showChat({ toEnd: !retry });
  showEvidence();
  try {
    const payload = { query, thread_id: run.thread.id, turn_id: turn.id,
      document_id: turn.documentId, settings: run.thread.settings,
      force_search: byId("force-search").checked };
    byId("force-search").checked = false;
    turn.result = await askStream(payload, (type, data) => {
      if (type === "start") {
        run.id = data.run_id;
        showComposer();
      } else if (type === "progress") {
        turn.steps = data;
      } else if (type === "evidence") {
        turn.evidenceActions = data;
        if (thread === run.thread) {
          const caption = byId("turn-" + turn.id)?.querySelector(".run-summary small");
          if (caption) caption.textContent = runSummary(turn);
          if (currentEvidence() === turn && evidenceTab === "trace") showEvidence();
        }
      }
    });
  } catch (error) {
    turn.result = interrupted(turn.steps, error.message);
    turn.evidenceActions = turn.evidenceActions.map((row) =>
      row.phase === "started" ? { ...row, phase: "interrupted" } : row);
  } finally {
    turn.steps = turn.result.steps;
    if (turn.result.evidence_actions) delete turn.evidenceActions;
    running = null;
    persist(run.thread);
    const nearBottom = chatScroll.scrollHeight - chatScroll.scrollTop - chatScroll.clientHeight < 30;
    showChat({ toEnd: nearBottom && !retry && thread === run.thread });
    showEvidence();
  }
}

async function stopRun() {
  const run = running;
  if (!run?.id || run.stopping) return;
  run.stopping = true;
  showComposer();
  try {
    await request("/api/runs/" + run.id + "/stop", { method: "POST" });
  } catch (error) {
    run.stopping = false;
    if (running === run) toast(error.message);
    showComposer();
  }
}

function suggest(query, scope = null) {
  showView("chat");
  question.value = query;
  thread.draftScope = scope;
  showComposer();
  question.focus();
}

function threadDialog(mode) {
  const deleting = mode === "delete";
  const modal = byId("thread-dialog");
  const input = byId("dialog-input");
  const confirm = byId("dialog-confirm");
  dialogAction = { mode, thread };
  byId("thread-menu").open = false;
  byId("dialog-title").textContent = deleting ? "删除对话" : "重命名对话";
  byId("dialog-description").hidden = !deleting;
  byId("dialog-label").hidden = deleting;
  input.required = !deleting;
  input.value = deleting ? "" : titleOf(thread);
  confirm.textContent = deleting ? "删除" : "保存";
  confirm.className = deleting ? "button danger" : "button primary";
  modal.showModal();
  if (deleting) modal.querySelector('[value="dismiss"]').focus();
  else input.select();
}

function exportThread() {
  byId("thread-menu").open = false;
  const turns = thread.turns.map((turn) => {
    const result = turn.result;
    const answer = result?.answer || result?.error || (result ? "已停止" : "检索中");
    const citations = (result?.sources ?? []).map((source) => {
      return "- " + source.source_label + " " + source.file_name + " · " + pageRange(source);
    });
    return ["## 问题", turn.query, "## 回答", answer, citations.join("\n")].filter(Boolean).join("\n\n");
  });
  const content = "# " + titleOf(thread) + "\n\n" + turns.join("\n\n---\n\n") + "\n";
  const blob = new Blob([content], { type: "text/markdown;charset=utf-8" });
  const link = element("a");
  link.href = URL.createObjectURL(blob);
  link.download = titleOf(thread).replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_").slice(0, 70) + ".md";
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

const library = initLibrary({
  onStatus(status, catalog, error) {
    const badge = byId("index-status");
    badge.dataset.status = status;
    badge.title = error ?? catalog?.index.collection_name ?? "";
    byId("index-status-text").textContent = { loading: "正在读取索引", ready: "索引已连接", error: "索引未就绪" }[status];
    byId("sidebar-index-text").textContent = catalog
      ? formatCount.format(catalog.documents.length) + " 篇文档 · " + formatCount.format(catalog.index.chunk_count) + " 个片段"
      : error ? "打开文档库查看详情" : "正在读取文档目录…";
  },
  askAbout(doc) {
    const { document_id, file_name } = doc;
    suggest("请根据《" + file_name + "》概括研究方法和主要结论，并标注引文。", { document_id, file_name });
  },
});
const agentSettings = initAgentSettings({ currentThread: () => thread, titleOf, persist });
byId("nav-settings").addEventListener("click", () => showView("settings"));

byId("new-chat").addEventListener("click", () => {
  openThread(thread.turns.length ? newThread() : thread);
  question.focus();
});
byId("thread-search").addEventListener("input", showHistory);
for (const id of ["nav-library", "open-library", "composer-library"]) {
  byId(id).addEventListener("click", () => showView("library"));
}
for (const id of ["nav-chat", "brand-home"]) byId(id).addEventListener("click", () => showView("chat"));
byId("open-history").addEventListener("click", () => {
  if (drawer === "history") {
    closeDrawer();
    return;
  }
  drawer = "history";
  chrome();
  byId("close-history").focus();
});
byId("toggle-evidence").addEventListener("click", () => {
  if (drawer === "evidence") closeDrawer();
  else if (!medium.matches && evidenceOpen) {
    evidenceOpen = false;
    chrome();
  } else {
    revealEvidence();
    showEvidence();
  }
});
for (const id of ["close-evidence", "close-history", "drawer-backdrop"]) byId(id).addEventListener("click", closeDrawer);
for (const media of [medium, phone]) media.addEventListener("change", () => { drawer = null; chrome(); showEvidence(); });
byId("tab-citations").addEventListener("click", () => setEvidenceTab("citations"));
byId("tab-trace").addEventListener("click", () => setEvidenceTab("trace"));
byId("stop-question").addEventListener("click", stopRun);
byId("clear-document-scope").addEventListener("click", () => {
  thread.draftScope = null;
  showComposer();
  question.focus();
});
document.querySelectorAll("[data-prompt]").forEach((card) => card.addEventListener("click", () => suggest(card.dataset.prompt)));
question.addEventListener("input", showComposer);
question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!byId("send-question").disabled) byId("composer").requestSubmit();
  }
});
byId("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  submit(question.value);
});
byId("rename-thread").addEventListener("click", () => threadDialog("rename"));
byId("delete-thread").addEventListener("click", () => threadDialog("delete"));
byId("export-thread").addEventListener("click", exportThread);
byId("dialog-form").addEventListener("submit", async (event) => {
  if (event.submitter?.value !== "confirm" || !dialogAction) return;
  const { mode, thread: item } = dialogAction;
  if (mode === "rename") {
    const title = byId("dialog-input").value.trim();
    if (!title) {
      event.preventDefault();
      byId("dialog-input").focus();
      return;
    }
    item.title = title;
    persist(item);
    showHistory();
    showChat();
  } else {
    event.preventDefault();
    const confirm = byId("dialog-confirm");
    confirm.disabled = true;
    try {
      await request("/api/conversations/" + item.id,
        { method: "DELETE", headers: { "X-EvidenceFlow-Workspace": "1" } });
      localStorage.removeItem(THREAD_KEY + item.id);
    } catch (error) {
      toast(error.message || "无法删除此对话。");
      return;
    } finally {
      confirm.disabled = false;
    }
    byId("thread-dialog").close();
    threads = threads.filter((saved) => saved !== item);
    openThread(threads[0] ?? newThread());
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    if (drawer) closeDrawer();
    byId("thread-menu").open = false;
  }
});
document.addEventListener("click", (event) => {
  if (!byId("thread-menu").contains(event.target)) byId("thread-menu").open = false;
});

openThread(thread);
library.load();
