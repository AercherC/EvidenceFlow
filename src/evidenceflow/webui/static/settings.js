/* Agent 设置保存在当前对话中，提交前由服务端校验并计算配套预算。 */
import { byId, element, json, postJSON } from "./ui.js";

export const BUDGET_LABELS = {
  searches: "检索轮数", opens: "原文阅读次数", deadline_seconds: "截止时间（秒）",
  decisions: "决策调用次数", llm_calls: "模型调用次数", rerank_pairs: "重排对数",
  reranker_input: "基础重排候选数", evidence_tokens: "证据预算（token）", reuse_evidence: "复用会话证据",
};

export function budgetValue(value) {
  return typeof value === "boolean" ? (value ? "开启" : "关闭") : String(value);
}

export function initAgentSettings({ currentThread, titleOf, persist }) {
  const form = byId("agent-settings-form");
  const fields = { searches: byId("setting-searches"), opens: byId("setting-opens"), deadline_seconds: byId("setting-deadline") };
  const message = byId("settings-message");
  const saveButton = byId("save-agent-settings");
  const resetButton = byId("reset-agent-settings");
  let defaults;
  let revision = 0;

  function notify(text) {
    message.textContent = text;
    message.hidden = !text;
  }

  function showBudget(effective) {
    const rows = [];
    for (const [key, title] of Object.entries(BUDGET_LABELS)) {
      const value = element("dd");
      value.append(element("span", "", budgetValue(effective[key])));
      if (effective[key] !== defaults[key]) {
        value.append(element("small", "settings-default", "服务默认 " + budgetValue(defaults[key])));
      }
      rows.push(element("dt", "", title), value);
    }
    byId("settings-effective").replaceChildren(...rows);
  }

  async function load() {
    const item = currentThread();
    const stamp = ++revision;
    byId("settings-thread").textContent = "当前对话 · " + titleOf(item);
    notify("");
    saveButton.disabled = true;
    resetButton.disabled = true;
    byId("settings-effective").replaceChildren();
    try {
      defaults ??= (await json("/api/agent/settings")).defaults;
      if (stamp !== revision) return;
      for (const [key, field] of Object.entries(fields)) {
        field.value = item.settings?.[key] ?? "";
        field.placeholder = String(defaults[key]);
      }
      byId("setting-memory").checked = item.settings?.reuse_evidence ?? defaults.reuse_evidence;
      const effective = item.settings ? await json("/api/agent/settings", postJSON(item.settings)) : defaults;
      if (stamp === revision) showBudget(effective);
    } catch (error) {
      if (stamp === revision) notify(error.message);
    } finally {
      if (stamp === revision) saveButton.disabled = resetButton.disabled = !defaults;
    }
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const item = currentThread();
    const stamp = revision;
    const settings = { reuse_evidence: byId("setting-memory").checked };
    for (const [key, field] of Object.entries(fields)) {
      if (field.value !== "") settings[key] = field.valueAsNumber;
    }
    saveButton.disabled = true;
    resetButton.disabled = true;
    try {
      const effective = await json("/api/agent/settings", postJSON(settings));
      item.settings = settings;
      persist(item);
      if (stamp === revision) {
        showBudget(effective);
        notify("已保存，从下一轮提问生效。");
      }
    } catch (error) {
      if (stamp === revision) notify(error.message);
    } finally {
      if (stamp === revision) saveButton.disabled = resetButton.disabled = false;
    }
  });
  resetButton.addEventListener("click", async () => {
    const item = currentThread();
    delete item.settings;
    persist(item);
    await load();
    if (currentThread() === item) notify("已恢复服务默认值。");
  });
  return { load };
}
