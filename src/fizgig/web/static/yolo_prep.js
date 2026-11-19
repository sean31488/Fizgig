// Fork-only (branch custom): a Custom YOLO choice on Image Prep's masks card, run by fizgig/dataset/yolo_prep.py.
// app.js calls yoloShow() at the end of maskSamShow; everything else lives here.
const yoloOn = () => !!(S.prepSaved || {}).maskYolo;
function yoloSave(patch) {
  S.prepSaved = {...(S.prepSaved || {}), ...patch};
  api("/api/state", {prep: S.prepSaved}).catch(() => {});
}
function yoloShow() {
  if (!$("yoloBox")) return;
  const on = yoloOn();
  $("maskCard").classList.toggle("yolo-on", on);
  $("maskMethod").querySelectorAll("button").forEach((b) => {
    if (b.id === "yoloBtn") b.classList.toggle("active", on); else if (on) b.classList.remove("active");
  });
  if (on) $("maskMethodHint").textContent = "Your own YOLO model (.pt, box or outline) through ultralytics, on the " +
    "GPU. Each picture and a close-up of each detection, all with their masks.";
  if (!$("yoloBox").dataset.filled) {
    const sv = S.prepSaved || {};
    $("yoloModel").value = sv.yoloModel || "";
    $("yoloPad").value = sv.yoloPad ?? 20;
    $("yoloBox").dataset.filled = "1";
  }
}
async function yoloRun() {
  const st = $("yoloState"), model = $("yoloModel").value.trim(), padIn = $("yoloPad").value, pad = padIn === "" ? 20 : Number(padIn);
  if (!S.imageFolder) { st.textContent = "Choose the training folder on the Start tab first."; return; }
  yoloSave({yoloModel: model, yoloPad: pad});
  if (!confirm("The pictures, captions and masks in the training folder are deleted and made again from its " +
               "originals subfolder. Go on?")) return;
  try {
    await api("/api/masks/yolo", {folder: S.imageFolder, model, pad});
    S.prep.masking = true;
    $("maskLog").textContent = ""; $("maskLog").style.display = "";
    $("prepLog").textContent = ""; S.prep.logSize = 0;
    st.textContent = "Started: see the log below.";
    pollPrep();
  } catch (e) { st.textContent = e.message; }
}
(() => {
  document.head.append(el("style", {text: "#maskCard.yolo-on .card-content > :not(.yolo-keep) {display: none !important}\n" +
    "#maskCard:not(.yolo-on) #yoloBox {display: none}"}));
  $("maskMethod").querySelectorAll("button[data-sam]").forEach((b) =>
    b.addEventListener("click", () => yoloSave({maskYolo: false})));
  $("maskMethod").append(el("button", {class: "btn", id: "yoloBtn", text: "Custom YOLO", onclick: () => {
    yoloSave({maskYolo: true}); if (S.prep.maskOpts) maskShow(S.prep.maskOpts); }}));
  $("maskMethodHint").after(el("div", {id: "yoloBox"},
    el("div", {class: "field-row"}, el("span", {class: "field-label", text: "YOLO model (.pt):"}),
      el("input", {type: "text", id: "yoloModel", class: "w-path", placeholder: "full path, e.g. D:\\models\\face_yolov8m-seg.pt"}),
      el("button", {class: "btn", text: "Choose…", onclick: () => pick($("yoloModel").value || S.imageFolder, false,
        (p) => { $("yoloModel").value = p; yoloSave({yoloModel: p}); })})),
    el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Close-up padding (%):"}),
      el("input", {type: "number", id: "yoloPad", min: "0", step: "5", value: "20"})),
    el("div", {class: "field-hint", text: "Pictures come from the originals subfolder (copied from this folder the " +
      "first time; add new pictures there). Every run deletes this folder's pictures, captions and masks and makes " +
      "them again. Needs ultralytics in Fizgig's venv: python -m pip install ultralytics"}),
    el("div", {class: "field-row"}, el("button", {class: "btn btn-primary", text: "Make pictures and masks", onclick: yoloRun}),
      el("span", {class: "field-hint", id: "yoloState"}))));
  const content = $("maskMethod").parentElement;
  for (const e of [content.querySelector(".mask-step"), $("maskMethod"), $("maskMethodHint"), $("yoloBox"),
                   $("maskInfo"), $("maskLog"), $("maskEdit")]) e.classList.add("yolo-keep");
})();
