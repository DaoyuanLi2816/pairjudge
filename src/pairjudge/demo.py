"""Loopback-only demo; no telemetry, remote assets or user-text logging."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd

HTML = r"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PairJudge · compare two answers</title><style>
:root{font-family:system-ui,sans-serif;color:#182d42;background:#f2f6fa}body{max-width:1000px;margin:auto;padding:32px 24px}h1{font-size:44px;letter-spacing:-2px;margin:0}header{border-bottom:1px solid #bfd0df;padding-bottom:20px;margin-bottom:24px}p{line-height:1.6;color:#425e73}.tag{color:#087f70;font-weight:650}label{display:block;font-weight:650;margin:16px 0 8px}textarea{box-sizing:border-box;width:100%;min-height:140px;border:1px solid #bfd0df;border-radius:12px;padding:14px;font:inherit;resize:vertical;background:white}#prompt{min-height:84px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}button{font:inherit;cursor:pointer;border:0;border-radius:9px;padding:12px 20px;background:#087f70;color:white;margin:18px 10px 18px 0}button.secondary{background:#dde7ee;color:#182d42}button:disabled{opacity:.6;cursor:wait}#status{white-space:pre-wrap}table{border-collapse:collapse;width:100%;background:white;border-radius:12px}th,td{text-align:left;padding:14px;border-bottom:1px solid #dde7ee}.bar{height:7px;background:#087f70;border-radius:5px;margin-top:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px;border-radius:12px;font-size:12px}#result{display:none}a{color:#087f70}@media(max-width:640px){.pair{grid-template-columns:1fr}h1{font-size:36px}body{padding:24px 16px}}
</style><header><span class="tag">PAIRWISE PREFERENCE · LOCAL INFERENCE</span><h1>Which answer would you prefer?</h1><p>Compare A, B and the learned tie category. Check both presentation orders with one trained judge.</p></header>
<p id="model">Loading model information…</p><label for="prompt">Prompt</label><textarea id="prompt" maxlength="20000"></textarea><div class="pair"><div><label for="a">Response A</label><textarea id="a" maxlength="20000"></textarea></div><div><label for="b">Response B</label><textarea id="b" maxlength="20000"></textarea></div></div>
<button id="run">Compare both orders</button><button id="example" class="secondary">Load example</button><button id="swap" class="secondary">Swap A / B</button><span id="status" role="status"></span>
<section id="result"><h2>Preference probabilities</h2><table><thead><tr><th>Outcome</th><th>Single pass</th><th>Swapped, aligned</th><th>Average</th></tr></thead><tbody id="scores"></tbody></table><p id="verdict"></p><p id="retention-note" class="tag"></p><details><summary>Retained content and truncation</summary><pre id="packing"></pre></details></section>
<p>This small example judge is uncalibrated and can be wrong. A numerical tie between probabilities is “ambiguous”; it is different from the learned tie class. Swap averaging gives consistent probabilities after exchanging A/B columns; it does not guarantee better accuracy.</p><p>Texts stay in this local server and are not uploaded or logged. Maximum: 60,000 characters, one inference request at a time. Long text may be cut; inspect the retention report. <a href="https://github.com/DaoyuanLi2816/pairjudge" target="_blank" rel="noopener">Documentation and evaluation</a></p>
<script>
const el=id=>document.getElementById(id); const example=()=>{el('prompt').value='Why does ice float on water? Explain in two sentences.';el('a').value='Ice forms an open crystal structure, so the same mass takes up more space than liquid water. Its lower density makes it float.';el('b').value='Ice floats because it is colder than water, and cold objects always rise.';};el('example').onclick=example;el('swap').onclick=()=>{const a=el('a').value;el('a').value=el('b').value;el('b').value=a;el('result').style.display='none';el('status').textContent='Responses swapped; compare again.';};example();
fetch('/metadata').then(r=>r.json()).then(m=>{el('model').textContent=`Model: ${m.model} · ${m.revision||'local bundle'} · ${m.device}/${m.dtype} · ${m.max_length} tokens`;});
el('run').onclick=async()=>{el('run').disabled=true;el('result').style.display='none';el('status').textContent='Comparing…';try{let r=await fetch('/compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:el('prompt').value,response_a:el('a').value,response_b:el('b').value})});let d=await r.json();if(!r.ok)throw Error(d.error);el('scores').replaceChildren();['a_wins','b_wins','tie'].forEach((key,i)=>{let tr=document.createElement('tr');let label=document.createElement('td');label.textContent=['A wins','B wins','Learned tie'][i];tr.append(label);[d.single,d.aligned,d.average].forEach(p=>{let td=document.createElement('td');td.textContent=(100*p[i]).toFixed(1)+'%';let bar=document.createElement('div');bar.className='bar';bar.style.width=(100*p[i])+'%';td.append(bar);tr.append(td)});el('scores').append(tr)});el('verdict').textContent='Average verdict: '+d.verdict+' · Single-pass order disagreement: '+d.disagreement;el('retention-note').textContent=d.packing.original.truncated||d.packing.swapped.truncated?'Input was truncated. Inspect retained content before using this verdict.':'All rounds fit the saved token budget.';el('packing').textContent=JSON.stringify(d.packing,null,2);el('result').style.display='block';el('status').textContent='Done';}catch(e){el('status').textContent=e.message}finally{el('run').disabled=false}};
</script></html>"""


def make_server(judge, port=7860):
    from .cli import input_record, run_metadata
    from .judge import swap_average
    from .validation import decisions

    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, data, content_type="application/json"):
            body = (
                data.encode("utf-8")
                if isinstance(data, str)
                else json.dumps(data).encode("utf-8")
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def valid_origin(self):
            allowed = {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }
            origin = self.headers.get("Origin")
            return self.headers.get("Host") in allowed and (
                origin is None or origin in {"http://" + host for host in allowed}
            )

        def do_GET(self):
            if not self.valid_origin():
                self.reply(403, {"error": "loopback origin required"})
            elif self.path == "/":
                self.reply(200, HTML, "text/html")
            elif self.path == "/metadata":
                self.reply(200, run_metadata(judge, True))
            else:
                self.reply(404, {"error": "not found"})

        def do_POST(self):
            if not self.valid_origin() or self.path != "/compare":
                self.reply(403, {"error": "loopback /compare required"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 180000:
                    # Bounded drain avoids a TCP reset for just-over-limit bodies.
                    if length > 0:
                        self.connection.settimeout(2)
                        try:
                            self.rfile.read(min(length, 180001))
                        except OSError:
                            pass
                    self.reply(413, {"error": "request body must be 1–180000 bytes"})
                    return
                self.connection.settimeout(10)
                record = input_record(json.loads(self.rfile.read(length)))
            except (ValueError, OSError) as exc:
                self.reply(400, {"error": str(exc)})
                return
            if not lock.acquire(blocking=False):
                self.reply(
                    429, {"error": "another comparison is running; retry shortly"}
                )
                return
            try:
                frame = pd.DataFrame([record])
                packed, swapped = (
                    judge._pack_frame(frame),
                    judge._pack_frame(judge._swap(frame)),
                )
                p, q = judge._forward(packed, 1), judge._forward(swapped, 1)
                avg = swap_average(p, q)
                self.reply(
                    200,
                    {
                        "single": p[0].tolist(),
                        "aligned": q[0, [1, 0, 2]].tolist(),
                        "average": avg[0].tolist(),
                        "verdict": decisions(avg)[0],
                        "disagreement": decisions(p) != decisions(q[:, [1, 0, 2]]),
                        "packing": {
                            "original": packed.iloc[0].diagnostics,
                            "swapped": swapped.iloc[0].diagnostics,
                        },
                    },
                )
            except (ValueError, RuntimeError) as exc:
                self.reply(400, {"error": str(exc)})
            finally:
                lock.release()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def serve(judge, port=7860):
    server = make_server(judge, port)
    print(
        f"PairJudge local demo: http://127.0.0.1:{server.server_port} (no text uploads)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
