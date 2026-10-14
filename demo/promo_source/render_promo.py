import asyncio, base64, json, os, subprocess, sys, time, urllib.request
import aiohttp
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PORT = 9340
MODE = sys.argv[1]
_t = open(f"{HERE}/promo_template.html", encoding="utf-8").read()
_d = open(f"{HERE}/data.json", encoding="utf-8").read().replace("</", "<" + chr(92) + "/")
open(f"{HERE}/promo.html", "w", encoding="utf-8").write(_t.replace("__DATA__", _d))
FPS = 30
DUR = 68.0
proc = subprocess.Popen([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1", f"--remote-debugging-port={PORT}",
                         f"--user-data-dir={HERE}/edge_render", "--window-size=1920,1080", "about:blank"])
for _ in range(40):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version", timeout=1); break
    except Exception:
        time.sleep(0.5)
ws_url = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/json/new?file:///{__import__("urllib.parse").parse.quote(HERE + "/promo.html", safe="/:")}", method="PUT")).read())["webSocketDebuggerUrl"]


async def main():
    async with aiohttp.ClientSession() as s, s.ws_connect(ws_url, max_msg_size=2 ** 28) as ws:
        n = [0]

        async def call(m, **p):
            n[0] += 1
            await ws.send_str(json.dumps({"id": n[0], "method": m, "params": p}))
            while True:
                r = json.loads((await ws.receive()).data)
                if r.get("id") == n[0]:
                    return r.get("result", r)
        await call("Emulation.setDeviceMetricsOverride", width=1920, height=1080, deviceScaleFactor=1, mobile=False)
        await asyncio.sleep(3)
        r = await call("Runtime.evaluate", expression="document.fonts.ready.then(() => [...document.fonts].filter(f => f.status === 'loaded').length)", awaitPromise=True, returnByValue=True)
        print("fonts loaded:", r["result"].get("value"), flush=True)
        if MODE == "samples":
            times = [float(x) for x in sys.argv[2].split(",")]
            names = [f"sample_{t:05.1f}.jpg" for t in times]
        else:
            os.makedirs(f"{HERE}/frames", exist_ok=True)
            times = [k / FPS for k in range(int(DUR * FPS))]
            names = [f"frames/f{k:05d}.jpg" for k in range(len(times))]
        t0 = time.time()
        for t, name in zip(times, names):
            await call("Runtime.evaluate", expression=f"render({t:.4f})")
            shot = await call("Page.captureScreenshot", format="jpeg", quality=94, clip={"x": 0, "y": 0, "width": 1920, "height": 1080, "scale": 1})
            open(f"{HERE}/{name}", "wb").write(base64.b64decode(shot["data"]))
            if MODE != "samples" and int(t * FPS) % 150 == 0:
                print(f"{t:.1f}s rendered, {time.time() - t0:.0f}s elapsed", flush=True)
        print("done", len(times), "frames", flush=True)

try:
    asyncio.run(main())
finally:
    proc.kill()
