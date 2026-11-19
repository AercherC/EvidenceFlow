/* 文档库从同一问答快照读取目录和片段，快速切换时取消旧的片段请求。 */
import { initImport } from "./library-import.js";
import { action, byId, element, emptyState, formatCount, icon, json, pageRange, passageCard } from "./ui.js";

export function initLibrary({ onStatus, askAbout }) {
  const items = byId("document-list");
  const search = byId("document-search");
  const reader = byId("document-reader");
  const refresh = byId("refresh-library");
  let catalog = null;
  let listing = false;
  let catalogError = "";
  let selected = null;
  let selectionRequest = null;
  let catalogRequest = null;

  function showList() {
    items.replaceChildren();
    const term = search.value.trim().toLowerCase();
    const docs = (catalog?.documents ?? []).filter((doc) => doc.file_name.toLowerCase().includes(term));
    byId("library-filter-count").textContent = catalog ? formatCount.format(docs.length) : "—";
    if (!docs.length) {
      let message = "当前索引没有可浏览的文档。";
      if (term && catalog?.documents.length) message = "没有找到匹配的文件。";
      if (catalogError) message = catalogError;
      if (listing) message = "正在读取文档目录…";
      items.append(element("li", "list-empty", message));
      return;
    }

    for (const doc of docs) {
      const file = element("span", "file-icon");
      file.append(icon("file"));
      const label = element("span");
      label.append(
        element("strong", "", doc.file_name),
        element("small", "", pageRange(doc) + " · " + formatCount.format(doc.chunk_count) + " 个片段"),
      );
      const button = action("", null, "document-item", () => {
        openPassage(doc, 0, true);
      });
      button.append(file, label, icon("chevron-right"));
      button.setAttribute("aria-pressed", String(selected?.document_id === doc.document_id));
      const row = element("li");
      row.append(button);
      items.append(row);
    }
  }

  function showReader(doc, offset, { passage, error, focus } = {}) {
    const header = element("div", "document-reader-header");
    const file = element("span", "file-icon");
    file.append(icon("file"));
    const title = element("h2", "document-title", doc.file_name);
    title.tabIndex = -1;
    header.append(file, title, element("p", "inline-note", "索引覆盖 · " + pageRange(doc)));

    const content = element("div", "document-reading");
    if (passage) content.append(passageCard(passage, "片段原文"));
    else if (error) {
      const hint = element("p", "inline-note", error);
      hint.setAttribute("role", "status");
      content.append(hint, action("重试", "refresh", "quiet-button", () => openPassage(doc, offset)));
    } else {
      const hint = element("p", "inline-note", "正在读取片段…");
      hint.setAttribute("role", "status");
      content.append(hint);
    }

    const pagination = element("div", "document-pagination");
    const previous = action("上一片段", "chevron-left", "button secondary", () => openPassage(doc, offset - 1));
    const next = action("下一片段", "chevron-right", "button secondary", () => openPassage(doc, offset + 1));
    previous.disabled = offset === 0;
    next.disabled = offset + 1 === doc.chunk_count;
    const position = element("span", "", (offset + 1) + " / " + doc.chunk_count);
    position.setAttribute("aria-label", "当前片段 " + (offset + 1) + "，共 " + doc.chunk_count + " 个");
    pagination.append(previous, position, next);
    content.append(pagination, action("就此文档提问", "messages", "quiet-button", () => askAbout(doc)));
    reader.replaceChildren(header, content);
    if (focus) title.focus();
  }

  async function openPassage(doc, offset, focus = false) {
    selected = doc;
    selectionRequest?.abort();
    const controller = new AbortController();
    selectionRequest = controller;
    showList();
    showReader(doc, offset, { focus });
    const params = new URLSearchParams({ document_id: doc.document_id, snapshot: catalog.snapshot, offset });
    try {
      const passage = await json("/api/passage?" + params, { signal: controller.signal });
      if (!controller.signal.aborted) {
        showReader(doc, offset, { passage, focus: reader.contains(document.activeElement) });
      }
    } catch (error) {
      if (!controller.signal.aborted) showReader(doc, offset, { error: error.message });
    }
  }

  function showEmptyReader() {
    reader.replaceChildren(emptyState("选择一篇文档", "打开左侧文件，可按索引顺序阅读片段。"));
  }

  async function loadCatalog() {
    listing = true;
    refresh.disabled = true;
    if (!catalog) onStatus("loading");
    showList();
    try {
      const updated = await json("/api/library");
      if (catalog?.snapshot !== updated.snapshot) {
        selected = null;
        selectionRequest?.abort();
        showEmptyReader();
      }
      catalog = updated;
      catalogError = "";
      byId("library-document-count").textContent = formatCount.format(catalog.documents.length);
      byId("library-chunk-count").textContent = formatCount.format(catalog.index.chunk_count);
      byId("library-collection").textContent = catalog.index.collection_name;
      onStatus("ready", catalog);
    } catch (error) {
      catalogError = error.message;
      catalog = null;
      selected = null;
      selectionRequest?.abort();
      for (const id of ["library-document-count", "library-chunk-count", "library-collection"]) {
        byId(id).textContent = "—";
      }
      showEmptyReader();
      onStatus("error", null, catalogError);
    } finally {
      listing = false;
      refresh.disabled = false;
      showList();
    }
  }

  function load() {
    if (!catalogRequest) {
      catalogRequest = loadCatalog().finally(() => { catalogRequest = null; });
    }
    return catalogRequest;
  }

  const imports = initImport(async (job) => {
    // 发布完成时，等此前可能已经开始的旧版本请求退出，再读取新索引。
    await catalogRequest;
    await load();
    const doc = catalog?.documents.find((item) => item.document_id === job.document_id);
    if (doc && !byId("library-screen").hidden) {
      search.value = "";
      await openPassage(doc, 0);
    }
  });

  search.addEventListener("input", showList);
  refresh.addEventListener("click", () => {
    load();
    imports.refresh();
  });
  showEmptyReader();
  return { load, checkImports: imports.refresh };
}
