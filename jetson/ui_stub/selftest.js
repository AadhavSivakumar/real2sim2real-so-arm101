// Keyboard-model + rendering self-test for web/index.html against ui_stub/stub.py.
// Injected by the stub with ?selftest=1; writes PASS/FAIL lines into <pre id="selftest-out"> (read via --dump-dom).
(async () => {
  const out = document.createElement("pre"); out.id = "selftest-out"; document.body.appendChild(out);
  const lines = [], posts = [], errors = [], alerts = [];
  let confirms = 0;
  window.addEventListener("error", (e) => errors.push(String(e.message)));
  window.addEventListener("unhandledrejection", (e) => errors.push("rejection: " + String(e.reason)));
  const realFetch = window.fetch.bind(window);
  window.fetch = (url, opts) => {
    if (opts && opts.method === "POST") posts.push({ path: String(url), body: JSON.parse(opts.body || "{}") });
    return realFetch(url, opts);
  };
  window.confirm = () => { confirms++; return true; };
  window.alert = (m) => alerts.push(m);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const $ = (s) => document.querySelector(s);
  const until = async (f, ms = 3000) => { const t = Date.now(); while (Date.now() - t < ms) { try { if (f()) return true; } catch (e) {} await sleep(50); } return false; };
  const phase = async (p, wantPhase, wantMode) => {
    await realFetch("/_phase?p=" + p);
    const ok = await until(() => state && state.policy.phase === wantPhase && (!wantMode || state.mode === wantMode));
    await sleep(250);  // a couple of renders
    if (!ok) lines.push(`FAIL scenario ${p} never reached phase ${wantPhase}`);
  };
  const key = (k, el, extra) => (el || document.activeElement || document.body)
    .dispatchEvent(new KeyboardEvent("keydown", Object.assign({ key: k, bubbles: true, cancelable: true }, extra || {})));
  const drain = async () => { await sleep(150); return posts.splice(0); };
  const paths = (ps) => ps.map((p) => p.path);
  const check = (name, cond, info) => lines.push(`${cond ? "PASS" : "FAIL"} ${name}${cond ? "" : "  :: " + JSON.stringify(info)}`);
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  const blur = () => { if (document.activeElement && document.activeElement.blur) document.activeElement.blur(); };

  try {
    // ---------------- ready (fresh page session)
    await phase("ready", "ready");
    await drain();
    blur(); key(" ");
    let ps = await drain();
    check("ready: Space before any click starts nothing", ps.length === 0, paths(ps));
    check("ready: Space explains why", /Start once|click Start/i.test($("#pflash").textContent), $("#pflash").textContent);
    check("ready: dry run defaults ON for the first run", $("#pdry").checked === true);
    check("ready: sidecar defaults applied", $("#pspeed").value === "120" && $("#plimit").value === "30" && $("#ptask").value === "Put the cube in the cup",
          [$("#pspeed").value, $("#plimit").value, $("#ptask").value]);
    $("#recstart").focus(); key(" ");
    ps = await drain();
    check("ready: Space on focused 'Start recording' does not submit the recording form", ps.length === 0, paths(ps));
    $("#pdry").focus(); key(" ");
    ps = await drain();
    check("ready: Space on focused Dry-run checkbox posts nothing and leaves it checked", ps.length === 0 && $("#pdry").checked, paths(ps));
    $("#pstart").click();
    ps = await drain();
    check("ready: Start click (dry) posts start dry_run=true without confirm",
          ps.length === 1 && ps[0].path === "/api/policy/start" && ps[0].body.dry_run === true && confirms === 0 && ps[0].body.record === null,
          { ps, confirms });
    check("ready: start body fields", ps[0] && same(Object.keys(ps[0].body).sort(), ["dry_run", "home_first", "max_speed", "n_action_steps", "record", "task", "time_limit_s"])
          && ps[0].body.max_speed === 120 && ps[0].body.time_limit_s === 30 && ps[0].body.n_action_steps === null && ps[0].body.home_first === true, ps[0] && ps[0].body);
    blur(); key(" ");
    ps = await drain();
    check("ready: Space after a dry click repeats a dry start", ps.length === 1 && ps[0].path === "/api/policy/start" && ps[0].body.dry_run === true, ps);
    $("#pdry").checked = false; $("#pdry").dispatchEvent(new Event("change"));
    blur(); key(" ");
    ps = await drain();
    check("ready: Space cannot start the first LIVE run", ps.length === 0, paths(ps));
    $("#pstart").click();
    ps = await drain();
    check("ready: first live Start asks for confirmation and posts dry_run=false",
          confirms === 1 && ps.length === 1 && ps[0].body.dry_run === false, { confirms, ps });
    blur(); key(" ");
    ps = await drain();
    check("ready: Space then repeats live start without another confirm", confirms === 1 && ps.length === 1 && ps[0].body.dry_run === false, { confirms, ps });
    $("#precds").value = "bad name!"; $("#pstart").click();
    ps = await drain();
    check("ready: invalid rollout dataset name is refused client-side", ps.length === 0 && /dataset name/.test($("#pflash").textContent), paths(ps));
    $("#precds").value = "rollouts_x"; $("#pnexec").value = "25"; $("#pstart").click();
    ps = await drain();
    check("ready: record + chunk steps go into the start body", ps.length === 1 && same(ps[0].body.record, { dataset: "rollouts_x" }) && ps[0].body.n_action_steps === 25, ps);
    $("#pnexec").value = "";
    $("#pdry").checked = true; $("#pdry").dispatchEvent(new Event("change")); await sleep(150);
    check("ready: record field disabled for dry runs", $("#precds").disabled === true);
    $("#pstart").click();
    ps = await drain();
    check("ready: dry start never sends record", ps.length === 1 && ps[0].body.record === null && ps[0].body.dry_run === true, ps);
    $("#pdry").checked = false; $("#pdry").dispatchEvent(new Event("change"));
    $("#ptask").focus();
    for (const k of ["1", "2", "0", "t", "h", " ", "Backspace"]) key(k);
    ps = await drain();
    check("ready: keys typed into the Task field do nothing", ps.length === 0, paths(ps));
    key("Escape");
    ps = await drain();
    check("ready: Esc inside a text field = STOP", same(paths(ps), ["/api/stop"]), paths(ps));
    key("Escape", null, { repeat: true });
    ps = await drain();
    check("ready: auto-repeated Esc does not spam", ps.length === 0, paths(ps));
    blur(); key("h");
    ps = await drain();
    check("ready: H = go to start pose", ps.length === 1 && ps[0].path === "/api/policy/home" && same(ps[0].body, { pose: null }), ps);
    key("t"); key("r"); key("1"); key("0"); key("Backspace");
    ps = await drain();
    check("ready: T/R/1/0/Backspace do nothing in ready", ps.length === 0, paths(ps));
    key("ArrowRight"); key("ArrowLeft");
    ps = await drain();
    check("ready: arrow keys do nothing outside a recording session", ps.length === 0, paths(ps));
    check("ready: no outcome prompt, no fault, no header pill", $("#pout").hidden && $("#pfault").hidden && $("#p-pol").hidden);
    check("ready: policy-view toggles visible", !$(".pvt").hidden);
    const cb = $('.pvt input[data-cam="wrist"]'); cb.checked = true; cb.dispatchEvent(new Event("change"));
    await sleep(300);
    check("ready: policy view switches the wrist stream to ?view=policy", /view=policy/.test($("#v-wrist img").src) && $("#v-wrist").classList.contains("pview"), $("#v-wrist img").src);
    check("ready: scene stream stays raw", !/view=policy/.test($("#v-scene img").src), $("#v-scene img").src);

    // ---------------- running (live)
    await phase("running", "running", "policy");
    await drain();
    check("running: header pill POLICY", /^POLICY \d+s$/.test($("#p-pol").textContent) && !$("#p-pol").hidden, $("#p-pol").textContent);
    check("running: disabled Policy mode button shown, Teleop/Manual not highlighted",
          !$('#modes [data-mode="policy"]').hidden && !document.querySelector("#modes button.on"));
    check("running: joint rows show policy targets, slider follows target",
          /policy/.test($("#row-elbow_flex .l").textContent) && Math.abs(+$("#row-elbow_flex input").value - state.policy.target.elbow_flex) < 0.51,
          [$("#row-elbow_flex .l").textContent, $("#row-elbow_flex input").value]);
    check("running: Success/Fail visible, Skip/Discard hidden", !$("#pout").hidden && $("#pskip").hidden && $("#pdisc").hidden);
    check("running: Start disabled, Stop enabled, Load/Unload disabled", $("#pstart").disabled && !$("#pstop").disabled && $("#pload").disabled && $("#punload").disabled);
    check("running: teleop 'Start recording' disabled", $("#recstart").disabled);
    blur(); key(" ");
    ps = await drain();
    check("running: Space = stop episode", same(paths(ps), ["/api/policy/stop"]), paths(ps));
    key("1"); ps = await drain();
    check("running: 1 = success (ends it)", ps.length === 1 && ps[0].path === "/api/policy/outcome" && ps[0].body.outcome === "success", ps);
    key("2"); ps = await drain();
    check("running: 2 = fail", ps.length === 1 && ps[0].body.outcome === "fail", ps);
    key("0"); key("Backspace"); key("r"); key("h");
    ps = await drain();
    check("running: 0/Backspace/R/H do nothing", ps.length === 0, paths(ps));
    key("T"); ps = await drain();
    check("running: T (shift) = take over", same(paths(ps), ["/api/policy/takeover"]), paths(ps));
    key("Escape"); ps = await drain();
    check("running: Esc = STOP (+ policy stop)", same(paths(ps).sort(), ["/api/policy/stop", "/api/stop"]), paths(ps));
    $("#pnotein").value = "cube slipped"; $("#pfail").click(); ps = await drain();
    check("running: note travels with the outcome", ps.length === 1 && same(ps[0].body, { outcome: "fail", note: "cube slipped" }), ps);
    $("#stop").click(); ps = await drain();
    check("running: header STOP button = /api/stop + /api/policy/stop", same(paths(ps).sort(), ["/api/policy/stop", "/api/stop"]), paths(ps));

    // ---------------- dry run
    await phase("running_dry", "running", "manual");
    check("dry: header pill DRY RUN", /^DRY RUN/.test($("#p-pol").textContent), $("#p-pol").textContent);
    check("dry: banner styled dry", $("#pstat").classList.contains("dry") && /DRY RUN/.test($("#pph").textContent));
    check("dry: ghost targets shown, arm sliders keep manual targets",
          $("#row-gripper .ghost").style.display === "block" && /dry-run policy/.test($("#row-gripper .pt").textContent) && /target/.test($("#row-gripper .l").textContent));
    check("dry: Policy mode button hidden", $('#modes [data-mode="policy"]').hidden);

    // ---------------- intervention
    await phase("intervention", "intervention", "teleop");
    await drain();
    blur(); key("r"); ps = await drain();
    check("intervention: R = resume", same(paths(ps), ["/api/policy/resume"]), paths(ps));
    key("t"); ps = await drain();
    check("intervention: T does nothing", ps.length === 0, paths(ps));
    key(" "); ps = await drain();
    check("intervention: Space = stop episode", same(paths(ps), ["/api/policy/stop"]), paths(ps));
    check("intervention: Resume visible, Take over hidden", !$("#presume").hidden && $("#ptake").hidden);

    // ---------------- pending
    await phase("pending", "pending");
    await drain();
    check("pending: amber prompt with all outcome buttons", !$("#pout").hidden && $("#pout").classList.contains("pending") && !$("#pskip").hidden && !$("#pdisc").hidden);
    check("pending: prompt mentions the unsaved rollout", /unsaved rollout/.test($("#pq").textContent), $("#pq").textContent);
    check("pending: header pill LABEL EPISODE", $("#p-pol").textContent === "LABEL EPISODE");
    blur(); key("0"); ps = await drain();
    check("pending: 0 = skip label (outcome none)", ps.length === 1 && same(ps[0].body, { outcome: "none" }), ps);
    key("Backspace"); ps = await drain();
    check("pending: Backspace = discard", same(paths(ps), ["/api/policy/discard"]), paths(ps));
    key("1"); ps = await drain();
    check("pending: 1 = success", ps.length === 1 && ps[0].body.outcome === "success", ps);
    key(" "); ps = await drain();
    check("pending: Space = start next (live was confirmed earlier)", ps.length === 1 && ps[0].path === "/api/policy/start" && ps[0].body.dry_run === false, ps);
    check("pending: start button says Start next", /Start next/.test($("#pstart").textContent), $("#pstart").textContent);

    // ---------------- fault (pending + fault)
    await phase("fault", "pending");
    check("fault: red fault box + POLICY FAULT pill", !$("#pfault").hidden && /tracking guard/.test($("#pfaultmsg").textContent) && $("#p-pol").textContent === "POLICY FAULT");

    // ---------------- homing
    await phase("homing", "homing", "policy");
    await drain();
    blur(); key(" "); ps = await drain();
    check("homing: Space = stop (+ STOP)", same(paths(ps).sort(), ["/api/policy/stop", "/api/stop"]), paths(ps));
    key("h"); ps = await drain();
    check("homing: H does nothing", ps.length === 0, paths(ps));

    // ---------------- teleop recording session active (+ a policy loaded)
    await phase("rec_session", "ready", "teleop");
    await drain();
    blur(); key(" "); ps = await drain();
    check("rec session: Space = recording next (not the policy)", same(paths(ps), ["/api/record/next"]), paths(ps));
    key("ArrowRight"); key("ArrowLeft"); ps = await drain();
    check("rec session: → next, ← rerecord", same(paths(ps), ["/api/record/next", "/api/record/rerecord"]), paths(ps));
    key("1"); key("h"); key("t"); key("Backspace"); ps = await drain();
    check("rec session: policy keys inactive", ps.length === 0, paths(ps));
    key("Escape"); ps = await drain();
    check("rec session: Esc = STOP + end session", same(paths(ps).sort(), ["/api/record/stop", "/api/stop"]), paths(ps));
    check("rec session: Start (policy) disabled with reason", $("#pstart").disabled && /recording session/.test($("#pnote").textContent));
    check("rec session: legend says keys go to the recording", /recording/.test($("#pkctx").textContent));

    // ---------------- limp
    await phase("limp", "ready", "limp");
    await drain();
    blur(); key(" "); key("h"); ps = await drain();
    check("limp: Space/H post nothing (Start/home disabled)", ps.length === 0, paths(ps));
    check("limp: reasons shown (torque + camera)", /torque is released/.test($("#pnote").textContent) && /scene camera/.test($("#pnote").textContent), $("#pnote").textContent);

    // ---------------- loading
    await phase("loading", "loading");
    const f = $("#recform"); f.dataset.value = ""; f.querySelector('[name=dataset]').value = "pick_cube"; f.querySelector('[name=task]').value = "Pick up the cube.";
    await drain();
    blur(); key(" "); ps = await drain();
    check("loading: Space does not submit the recording form", ps.length === 0, paths(ps));

    // ---------------- none: existing behaviour (Space starts a recording session)
    await phase("none", "none");
    await drain();
    check("none: policy view toggles hidden, wrist stream back to raw", $(".pvt").hidden && !/view=policy/.test($("#v-wrist img").src), $("#v-wrist img").src);
    blur(); key(" "); ps = await drain();
    check("none: Space submits the recording form (unchanged)", ps.length === 1 && ps[0].path === "/api/record/start" && ps[0].body.dataset === "pick_cube", ps);
    key("Escape"); ps = await drain();
    check("none: Esc = STOP", same(paths(ps), ["/api/stop"]), paths(ps));
    key("1"); key("h"); ps = await drain();
    check("none: policy keys inactive", ps.length === 0, paths(ps));

    // ---------------- error without a loaded policy: no policy -> recording keys rule applies
    await phase("error", "error");
    check("error: ERROR banner + fault", $("#pstat").classList.contains("error") && !$("#pfault").hidden);
    check("error: Load enabled, Start disabled", !$("#pload").disabled && $("#pstart").disabled);

    check("no alert() used", alerts.length === 0, alerts);
    check("no JS errors", errors.length === 0, errors);
  } catch (e) {
    lines.push("FAIL exception: " + (e && e.stack || e));
  }
  const fails = lines.filter((l) => l.startsWith("FAIL")).length;
  out.textContent = lines.join("\n") + `\nSUMMARY ${lines.length - fails} passed, ${fails} failed\nSELFTEST-DONE`;
})();
