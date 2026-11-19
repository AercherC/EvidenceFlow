/* 文档库的 PDF 上传、自动发布进度和异常恢复。 */
import { byId, element, json } from "./ui.js";

const WRITE_HEADERS = { "X-EvidenceFlow-Workspace": "1" };
const STEPS = [
  ["SCANNING", "读取 PDF"],
  ["PARSING", "解析文档"],
  ["CHUNKING", "清洗切块"],
  ["EMBEDDING", "向量化"],
  ["INDEXING", "写入索引"],
  ["VALIDATING", "确认写入"],
];
const TITLES = {
  BUILDING: "正在添加 PDF",
  READY: "索引写入完成，发布待恢复",
  PUBLISHING: "正在发布索引",
  PUBLISHED: "已发布至文档库",
  FAILED: "PDF 入库失败",
};
const WORKING = new Set(["BUILDING", "PUBLISHING"]);

export function initImport(onPublished) {
  const add = byId("add-pdf");
  const pdf = byId("pdf-file");
  const panel = byId("pdf-import");
  const stages = byId("import-stages");
  const publish = byId("publish-pdf");
  const discard = byId("discard-import");
  let state = null;
  let pending = "";
  let errorMessage = "";
  let timer;
  let version = 0;
  let publishedId;

  function show() {
    const job = state?.job;
    const status = job?.status;
    const title = pending || TITLES[status] || (state?.blocked ? "添加 PDF 暂不可用" : "添加 PDF");
    const showActions = status === "READY" || status === "FAILED";
    const error = errorMessage || job?.message || state?.blocked;
    const hint = error || (status === "BUILDING" ? "写入完成后自动更新文档库。" : "");
    add.disabled = Boolean(pending) || !state?.can_add;
    panel.hidden = !pending && !TITLES[status] && !hint;
    byId("import-title").textContent = title;
    const files = job?.file_progress;
    byId("import-file").textContent = (job?.file_name ?? "") + (files ? " · 已处理 " + files[0] + "/" + files[1] + " 个文件" : "");
    byId("import-message").textContent = hint;
    byId("import-message").hidden = !hint;
    byId("import-actions").hidden = !showActions;
    publish.hidden = status !== "READY";
    publish.disabled = Boolean(pending || state?.blocked);
    discard.disabled = Boolean(pending);

    stages.hidden = status !== "BUILDING";
    const current = Math.max(0, STEPS.findIndex(([phase]) => phase === job?.phase));
    stages.replaceChildren(...STEPS.map(([phase, label], index) => {
      const item = element("li", "", label);
      if (index < current) item.dataset.complete = "true";
      if (phase === STEPS[current][0]) item.setAttribute("aria-current", "step");
      return item;
    }));
  }

  async function refresh(keepError = false) {
    clearTimeout(timer);
    const stamp = ++version;
    try {
      const updated = await json("/api/imports/current");
      if (stamp !== version) return;
      state = updated;
      if (!keepError) errorMessage = "";
      if (state.job?.status === "PUBLISHED" && publishedId !== state.job.id) {
        publishedId = state.job.id;
        await onPublished(state.job);
      }
    } catch (error) {
      if (stamp !== version) return;
      errorMessage = error.message;
    }
    if (stamp === version) {
      show();
      if (WORKING.has(state?.job?.status)) timer = setTimeout(refresh, 1200);
    }
  }

  async function mutate(path, method, title, file) {
    if (pending) return;
    clearTimeout(timer);
    version += 1;
    errorMessage = "";
    pending = title;
    show();
    try {
      const options = { method, headers: WRITE_HEADERS };
      if (file) {
        options.body = new FormData();
        options.body.append("file", file);
      }
      const job = await json(path, options);
      state = { ...state, job };
    } catch (error) {
      errorMessage = error.message;
    } finally {
      pending = "";
      show();
      await refresh(true);
    }
  }

  add.addEventListener("click", () => pdf.click());
  pdf.addEventListener("change", () => {
    const [file] = pdf.files;
    pdf.value = "";
    if (file) mutate("/api/imports", "POST", "正在上传 PDF", file);
  });
  publish.addEventListener("click", () => {
    if (state?.job?.status === "READY") {
      mutate("/api/imports/" + state.job.id + "/publish", "POST", "正在发布索引");
    }
  });
  discard.addEventListener("click", () => {
    if (state?.job && window.confirm("放弃此次入库会移除本次上传的 PDF 和未发布候选。确定继续吗？")) {
      mutate("/api/imports/" + state.job.id, "DELETE", "正在清理候选");
    }
  });

  return { refresh };
}
