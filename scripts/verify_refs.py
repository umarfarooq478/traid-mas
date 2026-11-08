"""Look up candidate references by title on Crossref (published versions) and arXiv (preprints)."""
import truststore; truststore.inject_into_ssl()
import json, sys, time, urllib.parse, urllib.request, re, xml.etree.ElementTree as ET
UA = {"User-Agent": "traid-mas-refcheck/0.1 (academic reference verification)"}
def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode()
titles = [l.strip() for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
for t in titles:
    print("=== ", t)
    try:
        items = json.loads(get("https://api.crossref.org/works?rows=4&query.bibliographic=" + urllib.parse.quote(t)))["message"]["items"]
        for i in items:
            ti = (i.get("title") or [""])[0]
            if norm(ti)[:40] != norm(t)[:40]:
                continue
            au = "; ".join(f"{a.get('given','')} {a.get('family','')}".strip() for a in i.get("author", [])[:15])
            yr = (i.get("issued", {}).get("date-parts") or [[None]])[0][0]
            print("  CR", yr, "|", (i.get("container-title") or [""])[0], "|", ti, "| doi:", i.get("DOI"), "| pp", i.get("page", ""), "| vol", i.get("volume", ""), "| no", i.get("issue", ""), "|", au)
    except Exception as e:
        print("  CR ERR", e)
    try:
        x = get("http://export.arxiv.org/api/query?max_results=3&search_query=" + urllib.parse.quote('ti:"' + re.sub(r'[^\w\s\-]', ' ', t) + '"'))
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for e in ET.fromstring(x).findall("a:entry", ns):
            ti = " ".join(e.find("a:title", ns).text.split())
            if norm(ti)[:40] != norm(t)[:40]:
                continue
            au = "; ".join(a.find("a:name", ns).text for a in e.findall("a:author", ns)[:15])
            print("  AX", e.find("a:published", ns).text[:10], "|", e.find("a:id", ns).text, "|", ti, "|", au)
    except Exception as e:
        print("  AX ERR", e)
    time.sleep(3.5)
