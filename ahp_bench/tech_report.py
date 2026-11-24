"""Turkish technical report (<results>/report/TEKNIK_RAPOR.md), generated from the stored files.

CLI: python -m ahp_bench.tech_report --results STUDY/results --data STUDY/data [--report DIR]

Run after `python -m ahp_bench.report --results STUDY/results --data STUDY/data --tag main
--revision-tag revision_final`. Every number in the report is read or computed here from <results>/raw
(read-only, via ahp_bench.audit), from the report tables (<report>/tables/*.csv) and appendix files; none
is typed in.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np

from . import analysis as an
from . import audit, prompts, revise
from .config import JUDGE, MODELS, TEMPERATURE
from .prompts import condition_column as cond_col
from .prompts import condition_label as cond_label

COND_TR = {"lean": "yönlendirmesiz – doğrudan (direct)", "cot": "yönlendirmesiz – CoT", "guided": "yönlendirmeli (guided)"}
DIRECT = COND_TR["lean"]


# ---------------------------------------------------------------- helpers

def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))


def md(rows: list[dict], cols=None, rename=None) -> str:
    if not rows:
        return "(veri yok)"
    cols = cols or list(rows[0])
    rename = rename or {}
    esc = lambda s: str("" if s is None else s).replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(esc(rename.get(c, c)) for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    out += ["| " + " | ".join(esc(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def usd(v) -> str:
    return f"{float(v):.4f}"


def f4(v) -> str:
    return f"{float(v):.4f}"


def g(v) -> str:
    """Compact scientific form for very small differences."""
    v = float(v)
    return "0" if v == 0 else f"{v:.2e}"


def fence(text: str) -> list[str]:
    return ["```", text, "```"]


def table(tables: dict, tid: str) -> list[dict]:
    return tables.get(tid, [])


def panel(rows, key="Model") -> list[dict]:
    return [r for r in rows if r.get(key) == "Panel"]


# ---------------------------------------------------------------- build

def build(results_dir: Path, data_dir: Path, report_dir: Path) -> str:
    res = audit.run_audit(results_dir)
    exp = audit.expectations(res)
    raw = Path(results_dir) / "raw"
    calls = audit.load_jsonl(raw / "calls.jsonl")[0]
    revs = audit.load_jsonl(raw / "revisions.jsonl")[0]
    tables = {}
    for p in sorted((report_dir / "tables").glob("T*.csv")):
        tables[p.stem.split("_")[0]] = read_csv(p)
    inv = {(r["file"], r["run_tag"]): r for r in res["inventory"]}
    main_calls = [r for r in calls if r.get("run_tag") == audit.MAIN_TAG]
    rv = res["revision"]
    L = []
    add = L.append

    def ext(lines):
        L.extend(lines)

    # ------------------------------------------------------------ header
    n_fail = sum(1 for e in exp if e["status"] == "FAIL")
    ext(["# LLM–AHP deneyi: teknik rapor", "",
         "Bu rapor `python -m ahp_bench.tech_report` komutuyla kayıt dosyalarından üretilmiştir. Rapordaki sayılar "
         "`results/raw/` altındaki JSONL dosyalarından, `results/report/tables/` tablolarından ve "
         "`results/report/appendix/` eklerinden program tarafından okunmuş ya da hesaplanmıştır; elle yazılmamıştır. "
         "Ondalık ayırıcı olarak nokta kullanılmıştır. Zamanlar UTC'dir.", ""])

    # ------------------------------------------------------------ 1
    m = inv[("calls.jsonl", "main")]
    n_exp = len(an.load_expert_data(Path(data_dir) / "expert_matrices.json")["experts"])
    ext(["## 1. Amaç ve kapsam", "",
         f"Deneyde {len(MODELS)} büyük dil modeline (LLM) OpenRouter üzerinden tedarikçi değerlendirme ölçütleri için "
         "AHP ikili karşılaştırma soruları soruldu. Amaç, modellerin verdiği yanıtlardan AHP ağırlıklarını hesaplamak, "
         f"bunları aynı soruları yanıtlayan {n_exp} uzmanın ağırlıklarıyla karşılaştırmak ve ağırlıkların tedarikçi "
         "sıralamasına etkisini göstermektir.", "",
         f"Ana deney (`run_tag = main`) {m['records']} çağrıdan oluşur: {len(MODELS)} model × 3 koşul × 2 görev × "
         f"{audit.MAIN_REPS} tekrar. Ek olarak yönlendirmesiz – doğrudan (direct) koşul iki farklı sıcaklıkta tekrarlandı ve tutarsız matrisler için "
         "bir hakem modelle revizyon aşaması yürütüldü. Aşamaların kayıt tarihleri aşağıdadır.", ""])
    phases = [("Ön deneme (pilot, sonuçlara dahil değil)", "calls.jsonl", "pilot"),
              ("Ana deney", "calls.jsonl", "main")]
    phases += [(f"Sıcaklık kolu {t}", "calls.jsonl", t) for t in res["temp_tags"]]
    phases += [(f"Revizyon pilotu {t} (panel, dahil değil)", "revisions.jsonl", t)
               for t in ("revision_judge", "revision_v2", "rev_final", "rev_test_none") if ("revisions.jsonl", t) in inv]
    phases += [("Revizyon (revision_final), panel çağrıları", "revisions.jsonl", res["revision_tag"]),
               ("Revizyon (revision_final), hakem çağrıları", "judge_calls.jsonl", res["revision_tag"])]
    rows = [{"Aşama": lab, "Dosya": f, "run_tag": t, "Kayıt": inv[(f, t)]["records"],
             "İlk kayıt": inv[(f, t)]["first_utc"], "Son kayıt": inv[(f, t)]["last_utc"]}
            for lab, f, t in phases if (f, t) in inv]
    ext([md(rows), ""])

    # ------------------------------------------------------------ 2
    settings = {r["model"]: r for r in res["settings"] if r["file"] == "calls.jsonl" and r["run_tag"] == "main"}
    ext(["## 2. Deney tasarımı", "", "### 2.1 Modeller ve ayarlar", "",
         f"İstenen sıcaklık: {TEMPERATURE}. Akıl yürütme (reasoning) sağlayıcı izin verdiğinde kapatıldı "
         "(`effort=none`). Tablo, ana deney kayıtlarındaki istenen ve uygulanan değerleri gösterir.", ""])
    rows = []
    for k, cfg in MODELS.items():
        s = settings.get(k, {})
        rows.append({"Model": k, "Geliştirici": cfg["developer"], "OpenRouter kimliği": cfg["id"],
                     "Sıcaklık (istenen)": s.get("temperature_requested", ""),
                     "Sıcaklık (uygulanan)": s.get("temperature_applied", ""),
                     "Reasoning (uygulanan)": s.get("reasoning_applied", ""),
                     "Ort. reasoning token": "" if s.get("mean_reasoning_tokens") is None else f"{s['mean_reasoning_tokens']:.1f}",
                     "Sağlayıcılar (çağrı)": s.get("providers", ""), "Çağrı": s.get("records", "")})
    ext([md(rows), ""])
    pt = {(r["model"], r["extra_params"]): r for r in res["param_tests"]}
    gpt0 = next((r for (mk, e), r in pt.items() if mk == "gpt-5.2" and json.loads(e).get("temperature") == 0), None)
    gpt15 = next((r for (mk, e), r in pt.items() if mk == "gpt-5.2" and json.loads(e).get("temperature") == 1.5), None)
    mm_fail = [r for (mk, _), r in pt.items() if mk == "minimax-m2.5" and r["http_200"] == 0]
    ctrl = {r["model"] for r in res["param_tests"]} & {"gpt-4o", "claude-haiku-4.5"}
    notes = ["Notlar:", ""]
    notes.append("- gpt-4o için `reasoning` parametresi gönderilmedi (yapılandırmada kapalı; tabloda `omitted`).")
    notes.append(f"- minimax-m2.5 için reasoning zorunludur. `param_tests.jsonl` dosyasında reasoning'i kapatmaya "
                 f"çalışan {len(mm_fail)} istek HTTP 400 ile reddedildi (hata metni: "
                 f"\"{mm_fail[0]['errors'].rsplit(' (', 1)[0] if mm_fail else ''}\"). Ana deneyde en düşük düzey "
                 f"`effort=minimal` uygulandı.")
    if gpt0:
        notes.append(f"- gpt-5.2 sıcaklık parametresini kabul eder ama uygulamaz. `param_tests.jsonl` dosyasında "
                     f"T = 0 ile yapılan {gpt0['calls']} çağrı {gpt0['distinct_answer_sets']} farklı yanıt kümesi verdi"
                     + (f"; T = 1.5 ile yapılan {gpt15['calls']} çağrı {gpt15['distinct_answer_sets']} farklı yanıt kümesi verdi"
                        if gpt15 else "") + ".")
    if not ctrl:
        notes.append("- Aynı sınamanın claude-haiku-4.5 ve gpt-4o ile yapılan karşılaştırma çağrıları bu dosyaya "
                     "kaydedilmedi; bu karşılaştırma yalnızca oturum içi bir kontroldür ve bu raporda sayısal "
                     "kanıt olarak kullanılmamıştır.")
    ext(notes + [""])

    # conditions
    pt_tok = {}
    for r in main_calls:
        pt_tok.setdefault((r["condition"], r["task"]), []).append((r.get("usage") or {}).get("prompt_tokens") or 0)
    gpay = prompts.guided_payload("main")
    gpay_d = prompts.guided_payload("delivery")
    cot_add = prompts.COT_FORMAT
    lean_user = prompts.build_messages("lean", "main")[1]["content"]
    assert prompts.build_messages("cot", "main")[1]["content"] == lean_user.replace(prompts.LEAN_FORMAT, cot_add)
    ext(["### 2.2 Koşullar", "",
         "Koşullar iki eksende tanımlanır: istem tekniği (sıfır atış = örnek çözüm ve akıl yürütme talimatı yok; "
         "CoT = adım adım düşünme talimatı) ve ağırlık yönlendirmesi (yönlendirmeli = istemde öncelik sırası ve önem "
         "ipuçları var; yönlendirmesiz = tarafsız sorular). Üç koşul çalıştırıldı; CoT × yönlendirmeli hücresi "
         "çalıştırılmadı. Kısa adlar: guided, direct (kayıt anahtarı `lean`), CoT (kayıt anahtarı `cot`).", "",
         "| Teknik / yönlendirme | Yönlendirmeli | Yönlendirmesiz |", "| --- | --- | --- |",
         "| Sıfır atış | yönlendirmeli (guided): `guided` | yönlendirmesiz – doğrudan (direct): `lean` |",
         "| CoT | çalıştırılmadı | yönlendirmesiz – CoT: `cot` |", "",
         "| İstem içeriği | Yönlendirmeli (guided) | Yönlendirmesiz – doğrudan (direct) | Yönlendirmesiz – CoT |",
         "| --- | --- | --- | --- |",
         "| Rol satırı | tedarikçi değerlendirmesinde uzman AHP uzmanı, AHP kurallarıyla | tedarikçi değerlendirme "
         "deneyimli satın alma uzmanı | doğrudan ile aynı |",
         "| Öncelik sırası | var: tipik öncelik sırası (Kalite en yüksek … Hacim düşük); “never use all 1s” | yok | yok |",
         "| Ölçüt ipuçları | var: önem bildiren Türkçe ölçüt açıklamaları | yok (yalnız adlar) | yok (yalnız adlar) |",
         "| KPI raporu | var: tedarikçilere ait Türkçe veri raporu | yok | yok |",
         "| Akıl yürütme talimatı | her karşılaştırma için gerekçe istenir | yok | adım adım düşün, önce sıralamayı yaz |",
         "| Yanıt biçimi | etiketli tam ikili karşılaştırma matrisi ve gerekçeler (JSON) | soru başına bir değer (JSON) | "
         "gerekçe, sıralama ve soru başına bir değer (JSON) |", "",
         "Doğrudan ve CoT koşullarında model her soru için tek bir değer verir ve matrisi kod kurar; guided koşulunda "
         "model matrisi kendisi döndürür.", "",
         f"**{COND_TR['lean']}.** Kayıtlarda bu koşulun anahtarı `lean`'dir (istem kimlikleri ve özetler bu "
         "anahtarla tutulur). Sistem mesajı:", ""] + fence(prompts.LEAN_SYSTEM) + [
         "", "Kullanıcı mesajı; ölçütlerin kısa tanıtımını, 17 düzeyli Saaty ölçeğini (9 … 1 … 1/9), soru listesini "
         "ve yalnızca `answers` alanını isteyen JSON biçimini içerir. Ölçütler için tanım, öncelik ipucu ya da veri "
         "verilmez.", "",
         "**Yönlendirmesiz – CoT (adım adım düşünme).** CoT de ağırlıklar açısından yönlendirmesizdir: doğrudan isteme "
         "yalnız adım adım düşünme talimatı ekler (kayıtlarda anahtarı `cot`). Sistem mesajı, ölçek ve sorular yönlendirmesiz – doğrudan (direct) koşulla aynıdır. Yanıt biçiminin yerine şu "
         "metin gelir:", ""] + fence(cot_add) + [
         "", "**Yönlendirmeli (guided).** Tezde kullanılan özgün istem `prompts/guided/` dosyalarından bayt bayt "
         "aynen gönderilir (denetim d). Sistem mesajı AHP kurallarını ve “never use all 1s” kuralını içerir. "
         f"Ana ölçüt kullanıcı mesajı bir JSON nesnesidir; üst düzey alanları: {', '.join(f'`{k}`' for k in gpay)}. "
         f"`instructions` alanının alt alanları: {', '.join(f'`{k}`' for k in gpay['instructions'])}. Bu istem bir "
         "öncelik sırası (`priority_guidance`), ipucu taşıyan ölçüt açıklamaları (`main_criteria[].description`) ve "
         "`scenario_context` içinde Türkçe tedarikçi KPI raporunu içerir. Teslimat alt ölçüt mesajının alanları: "
         f"{', '.join(f'`{k}`' for k in gpay_d)}.", "",
         "Ana deneydeki ortalama istem (prompt) token sayısı:", ""])
    rows = [{"Koşul": cond_label(c), "Görev": t, "Çağrı": len(pt_tok.get((c, t), [])),
             "Ort. prompt token": f"{np.mean(pt_tok[(c, t)]):.1f}" if pt_tok.get((c, t)) else ""}
            for c in prompts.CONDITIONS for t in prompts.TASKS]
    ext([md(rows), ""])
    ext(["### 2.3 Görevler", "",
         f"- `main`: {len(prompts.MAIN_CRITERIA)} ana ölçüt, {len(prompts.PAIRS['main'])} soru: "
         + "; ".join(prompts.MAIN_CRITERIA) + ".",
         f"- `delivery`: Teslimat ve operasyonel performansın {len(prompts.DELIVERY_SUBCRITERIA)} alt ölçütü, "
         f"{len(prompts.PAIRS['delivery'])} soru: " + "; ".join(prompts.DELIVERY_SUBCRITERIA) + ".",
         "- Soru sırası uzmanlara sorulan sorulardaki sıradır: (1,2), (1,3), …, (n−1,n).", "",
         "### 2.4 Tekrarlar ve sıcaklık kolları", "",
         f"- Ana deney: her model × koşul × görev hücresi için {audit.MAIN_REPS} bağımsız çağrı, sıcaklık {TEMPERATURE}.",
         ] + [f"- `{t}`: yalnız yönlendirmesiz – doğrudan (direct) koşul, {audit.TEMP_REPS} tekrar, sıcaklık {t.split('_')[1]}; "
              f"{inv[('calls.jsonl', t)]['records']} çağrı." for t in res["temp_tags"]] + [
         f"- `pilot` etiketli {inv[('calls.jsonl', 'pilot')]['records']} çağrı ön denemedir ve sonuçlara alınmadı."
         if ("calls.jsonl", "pilot") in inv else "", ""])

    # ------------------------------------------------------------ 3
    comp = {c["tag"]: c for c in res["completeness"]}
    fc = res["fields"]
    ext(["## 3. Kayıt (loglama) ve doğrulanabilirlik", "", "### 3.1 Ne kaydedildi, nerede", "",
         "Her API çağrısı tek bir JSON satırı olarak eklenir; dosyalar yalnızca sona ekleme ile büyür.", "",
         "- `results/raw/calls.jsonl`: ana deney, sıcaklık kolları ve pilot çağrıları.",
         "- `results/raw/revisions.jsonl`: revizyon aşamasındaki panel modeli çağrıları.",
         "- `results/raw/judge_calls.jsonl`: hakem modeli çağrıları (geri bildirim ve karar).",
         "- `results/raw/param_tests.jsonl`: parametre sınamaları (gpt-5.2 sıcaklığı, minimax-m2.5 reasoning).",
         "- `results/*.log`: çalıştırma sırasında ekrana yazılan ilerleme satırları.", "",
         "Her çağrı kaydındaki alanlar: gönderilen mesajların tamamı (`messages`), istem kimliği ve SHA-256 özeti "
         "(`prompt_id`, `prompt_sha256`), istenen ve uygulanan parametreler (`requested_params`, `applied_params`, "
         "atlanan parametreler `omitted`), her HTTP denemesi (`http_attempts`), ham yanıt (`raw_reply`), modelin "
         "akıl yürütme metni varsa (`model_reasoning_text`), bitiş nedeni, token kullanımı (prompt, completion, "
         "reasoning), maliyet, sağlayıcı, OpenRouter üretim kimliği (`generation_id`), gecikme ve zaman damgası. "
         "Ayrıştırma sonucunda yanıtlar, matris, ağırlıklar, λmax, CI ve CR aynı kayda yazılır. API anahtarı "
         "kayda girmeden önce maskelenir (`client._scrub`).", "",
         "### 3.2 Denetim sonuçları", "",
         f"`python -m ahp_bench.audit` {len(exp)} beklenti denetimi yaptı; {len(exp) - n_fail} tanesi geçti, "
         f"{n_fail} tanesi kaldı. Ayrıntılar `results/report/audit/audit.md` ve aynı klasördeki CSV dosyalarındadır.", ""])
    if n_fail:
        ext([md([e for e in exp if e["status"] == "FAIL"]), ""])
    inv_rows = [{"Dosya": r["file"], "run_tag": r["run_tag"], "Kayıt": r["records"], "Başarılı çağrı": r["calls_ok"],
                 "Başarısız": r["calls_failed"], "parse_ok": "" if r["parse_ok"] is None else r["parse_ok"],
                 "Maliyet (USD)": usd(r["cost_usd"]), "Prompt tok.": r["prompt_tokens"],
                 "Completion tok.": r["completion_tokens"], "Reasoning tok.": r["reasoning_tokens"]}
                for r in res["inventory"] if not r["backup"]]
    ext(["**a. Envanter.** Dosya ve etiket başına kayıt, başarı, ayrıştırma, maliyet ve token sayıları:", "",
         md(inv_rows), ""])
    ext([f"**b. Tasarımın tamlığı.** `main` etiketinde beklenen {comp['main']['expected']} hücrenin "
         f"{comp['main']['cells_present']} tanesi kayıtlıdır; eksik {len(comp['main']['missing'])}, yinelenen "
         f"{len(comp['main']['duplicated'])}, beklenmeyen {len(comp['main']['unexpected'])} hücre vardır; ayrıştırılmış "
         f"kaydı olmayan hücre sayısı: {len(comp['main']['cells_without_parse_ok'])}. "
         + " ".join(f"`{t}` için beklenen {comp[t]['expected']}, kayıtlı {comp[t]['cells_present']}, eksik "
                    f"{len(comp[t]['missing'])}, yinelenen {len(comp[t]['duplicated'])}." for t in res["temp_tags"]), ""])
    fc_rows = [{"Dosya": r["file"], "run_tag": r["run_tag"], "Kayıt": r["records"],
                **{k: ("—" if r[k] is None else r[k]) for k in audit.FIELD_CHECKS},
                "Eksik alanlı kayıt": r["records_lacking_any"]} for r in fc]
    ext(["**c. Alan tamlığı.** Sonuç etiketlerindeki her kayıtta alanın dolu olduğu kayıt sayısı "
         "(`usage_tokens`: prompt ve completion token tamsayı; hakem kayıtlarında istem kimliği tutulmaz, “—”):", "",
         md(fc_rows), ""])
    pi = res["prompt_integrity"]
    gf = res["guided_files"]
    main_pi = [p for p in pi if p["run_tag"] == "main"]
    ext([f"**d. İstem bütünlüğü.** `main` etiketinde {len(main_pi)} koşul × görev çiftinin her birinde farklı "
         f"`prompt_sha256` sayısı: {', '.join(str(p['distinct_prompt_sha256']) for p in main_pi)}. Saklanan özet ile "
         f"saklanan mesajların yeniden hesaplanan özeti {sum(p['stored_hash_matches_stored_messages'] for p in main_pi)}/"
         f"{sum(p['records'] for p in main_pi)} kayıtta aynıdır; saklanan mesajlar bugünkü kodun ürettiği mesajlarla "
         f"{sum(p['messages_equal_current_code'] for p in main_pi)}/{sum(p['records'] for p in main_pi)} kayıtta aynıdır. "
         + " ".join(f"guided/{x['task']}: sistem mesajı dosyayla (CRLF→LF) {x['system_message_equals_file']}/{x['records']}, "
                    f"kullanıcı mesajı dosya baytlarıyla {x['user_message_equals_file_bytes']}/{x['records']} kayıtta aynı."
                    for x in gf)
         + f" Revizyon kayıtlarında saklanan özet ile mesaj özeti {res['revision_prompt_hash'][0]['stored_hash_matches_stored_messages']}/"
           f"{res['revision_prompt_hash'][0]['records']} kayıtta aynıdır.", ""])
    multi = [r for r in settings.values() if ";" in r["providers"]]
    fb_calls = sum(r["calls_with_parameter_fallback"] for r in settings.values())
    ext([f"**e. Uygulanan ayarlar.** Ana deneyde istenen ve uygulanan sıcaklık her modelde aynıdır (2.1'deki tablo). "
         f"Parametre düşürme (reasoning ya da sıcaklığın geri çekilmesi) gerektiren çağrı sayısı: {fb_calls}. "
         f"{len(multi)} model birden fazla sağlayıcıya yönlendirildi: "
         + "; ".join(f"{r['model']}: {r['providers']}" for r in multi) + ". Ana deney çağrılarının bitiş nedenleri: "
         + audit._counter_text(Counter(str(r.get("finish_reason")) for r in main_calls)) + ".", ""])
    pv = [r for r in res["parser_versions"] if r["file"] == "calls.jsonl"]
    no_pv = sorted(audit._ts(r).isoformat(timespec="seconds") for r in main_calls if "parser_version" not in r)
    flags = {(r["condition"], r["task"]): r for r in res["parse_flags"] if r["file"] == "calls.jsonl" and r["run_tag"] == "main"}
    bk = res["backups"]
    gl = [flags[("guided", t)] for t in prompts.TASKS if ("guided", t) in flags]
    ext(["**f. Ayrıştırma.** Ayrıştırıcı ana deneyden sonra iyileştirildi ve saklanan ham yanıtlar yeniden "
         "ayrıştırıldı (`parser_version`). Kayıtların ayrıştırıcı sürümüne göre dağılımı (`absent`: alan yok):", "",
         md(pv, ["run_tag", "parser_version", "records"]), "",
         f"`main` etiketinde `parser_version` alanı olmayan {len(no_pv)} kaydın ilk zaman damgası {min(no_pv) if no_pv else ''}; "
         f"yeniden ayrıştırma öncesi yedekteki son `main` kaydının zaman damgası "
         f"{inv.get(('calls_backup_before_reparse.jsonl', 'main'), {}).get('last_utc', '')}. Bu kayıtlar yeniden ayrıştırmadan "
         "sonra yapılan çağrılardır ve çağrı anında ayrıştırılmıştır; g bölümündeki yeniden ayrıştırma bunlar için de "
         "aynı sonucu verir.", "",
         "Guided yanıtlarında etiket ve karşılıklılık durumu (ana deney):", "",
         md(gl, ["task", "records", "parse_ok", "labels_verified", "labels_not_verified", "reordered",
                 "reciprocal_as_returned", "not_reciprocal_as_returned", "upper_on_scale", "bare_fractions_quoted"]), "",
         "`labels_verified`: döndürülen etiketler ölçütlerle bire bir eşlendi; `reordered`: model ölçütleri farklı "
         "sırada verdi ve matris kanonik sıraya çevrildi; `reciprocal_as_returned`: döndürülen matris köşegeni 1 ve "
         "a_ij·a_ji = 1 (±0.01); analizde her durumda üst üçgenden karşılıklı matris kurulur; `bare_fractions_quoted`: "
         "tırnaksız kesirler (1/3) JSON için tırnağa alındı. Yönlendirmesiz – doğrudan (direct) ve CoT yanıtlarında bu işlem "
         f"{sum(flags[(c, t)]['bare_fractions_quoted'] for c in ('lean', 'cot') for t in prompts.TASKS)} kez gerekti.", "",
         "Yedek dosyalar (`calls_backup_before_reparse.jsonl`, `calls.jsonl.bak`) üretim kimliğiyle eşlendi; ham alanlar "
         "(ham yanıt, mesajlar, kullanım, maliyet, sağlayıcı, zaman, parametreler) "
         + "; ".join(f"{b['backup']}: {b['raw_fields_identical']}/{b['backup_records']} kayıtta aynı" for b in bk)
         + ". Yani ham yanıtlar yeniden ayrıştırmada değişmedi. Bu kayıtlarda ayrıştırılan kayıt sayısı yedekte "
         + ", ".join(f"{b['parse_ok_backup']} ({b['backup']})" for b in bk)
         + f"; aynı kayıtlarda bugün: {bk[0]['parse_ok_now_same_records'] if bk else ''}.", ""])
    rc = res["recompute"]
    rp = res["reparse"]
    worst = max(max(v for k, v in r.items() if k.startswith("max_abs_diff_")) for r in rc)
    ext(["**g. Yeniden hesaplama.** Her ayrıştırılmış kayıt için matris saklanan yanıtlardan (direct, CoT) ya da "
         "döndürülen matristen (guided) yeniden kuruldu; ağırlık, λmax, CI ve CR `ahp` fonksiyonlarıyla yeniden "
         f"hesaplandı. Saklanan değerlerle en büyük mutlak fark: {g(worst)}.", "",
         md([{"Dosya": r["file"], "run_tag": r["run_tag"], "Kayıt": r["recomputed"],
              "Matris": g(r["max_abs_diff_matrix"]), "Ağırlık": g(r["max_abs_diff_weights"]),
              "λmax": g(r["max_abs_diff_lambda_max"]), "CI": g(r["max_abs_diff_CI"]), "CR": g(r["max_abs_diff_CR"])}
             for r in rc]), "",
         "Ayrıca tüm ham yanıtlar bugünkü ayrıştırıcıyla dosyaya yazmadan yeniden ayrıştırıldı: "
         + "; ".join(f"{r['file']} {r['run_tag']}: parse_ok uyumu {r['parse_ok_agrees']}/{r['records_with_reply']}, "
                     f"bayrak farkı {r['records_with_flag_difference']}, en büyük CR farkı {g(r['max_abs_diff_CR'])}"
                     for r in rp if r["run_tag"] in ("main", "revision_final") or r["run_tag"].startswith("temp_")) + ".", ""])
    sec = res["secrets"]
    hits = [s for s in sec if s["key_prefix_count"] or s["bearer_count"]]
    ext([f"**h. Gizli bilgi taraması.** {', '.join(f'`{d}/`' for d in res['scan_dirs'])} altındaki {len(sec)} dosya "
         "(`__pycache__` hariç) OpenRouter anahtar öneki ve `Bearer` sözcüğü için tarandı. Anahtar biçimindeki "
         f"eşleşme sayısı: {sum(s['key_like_count'] for s in sec)}; `Bearer` sözcüğünden sonra belirteç biçiminde dize "
         f"sayısı: {sum(s['bearer_token_like_count'] for s in sec)}. Anahtar önekini içeren dosya sayısı: "
         f"{sum(1 for s in sec if s['key_prefix_count'])}. `Bearer` sözcüğünü içeren dosyalar (içerik yazdırılmadı): "
         + (", ".join(f"`{s['file']}` ({s['bearer_count']})" for s in hits if s["bearer_count"]) or "yok") + ". "
         "`.env` dosyası taranan klasörlerin dışındadır ve okunmadı.", ""])
    ext(["**i. Sağlama toplamları.** `results/raw` dosyalarının boyutu ve SHA-256 özeti:", "",
         md([{"Dosya": r["file"], "Bayt": r["bytes"], "Satır": r["lines"], "SHA-256": r["sha256"]}
             for r in res["checksums"]]), ""])
    ext(["### 3.3 Nasıl kontrol edilir", "",
         "Kod `https://github.com/baa-git/llm-ahp-mcdm` deposundadır (sürüm v0.2.0); kayıtlar, veriler ve bu rapor "
         "`https://github.com/baa-git/llm-ahp-mcdm-study` deposundadır. Komutlar kod deposunun kökünde çalıştırılır; "
         "`<study>` çalışma deposunun yerel klasörüdür. Hiçbiri API çağrısı yapmaz.", "",
         "1. `python -m unittest discover -s tests -v` — Çevrimdışı birim testleri (sahte HTTP katmanı ve yapay "
         "veri). Ayrıştırıcının, AHP hesaplarının, revizyon protokolünün, raporun ve denetimin beklenen davranışı "
         "verdiğini gösterir.",
         "2. `python -m ahp_bench.audit --results <study>/results` — Bu bölümdeki a–j denetimlerini saklanan "
         "dosyalar üzerinde yeniden yapar; `results/raw` dosyalarını yalnızca okur. Tamlığı, istem bütünlüğünü, "
         "yeniden hesaplamayı, gizli bilgi yokluğunu ve revizyon kurallarına uyumu gösterir.",
         "3. `python -m ahp_bench.report --results <study>/results --data <study>/data --tag main "
         "--revision-tag revision_final` — Tüm tabloları, şekilleri ve ekleri kayıtlardan yeniden üretir. Çıktı "
         "belirlenimlidir (aynı girdiyle aynı tablolar).",
         "4. `python -m ahp_bench.tech_report --results <study>/results --data <study>/data` — Bu raporu yeniden üretir.",
         "5. `python -m ahp_bench.reparse --results <study>/results` — Ham yanıtları bugünkü ayrıştırıcıyla yeniden "
         "ayrıştırır ve `calls.jsonl` içindeki türetilmiş alanları yeniden yazar (önce `.bak` kopyası alır; ham "
         "alanlara dokunmaz). Dosyayı değiştirdiği için yalnız doğrulama amacıyla gerekmez: denetimin g bölümü "
         "aynı karşılaştırmayı dosyaya yazmadan yapar.",
         "6. Sağlama toplamı: `certutil -hashfile results\\raw\\calls.jsonl SHA256` (Windows) ya da "
         "`sha256sum results/raw/*` çıktısı 3.2-i tablosuyla karşılaştırılır. Aynı özet, dosyanın bu rapordan "
         "sonra değişmediğini gösterir.", ""])

    # ------------------------------------------------------------ 4
    perf = an.load_performance(Path(data_dir) / "performance_matrix.json")
    ben = ", ".join(f"{c} ({'fayda' if b else 'maliyet'})" for c, b in zip(perf["codes"], perf["benefit_mask"]))
    ext(["## 4. Hesaplama kuralları", "",
         "- **Yanıtlardan matris.** k. soru (i, j) çiftine aittir. Değer v ise a_ij = v, a_ji = 1/v, a_ii = 1. "
         "Guided koşulunda model matrisi döndürür; etiketler kanonik sıraya eşlenir ve yalnız üst üçgen kullanılır "
         "(a_ji = 1/a_ij).",
         "- **Ağırlıklar.** A·w = λmax·w denkleminin en büyük gerçel özdeğerine ait sağ özvektör; |w| alınır ve "
         "Σw = 1 olacak biçimde normalleştirilir (`numpy.linalg.eig`).",
         f"- **Tutarlılık.** CI = (λmax − n)/(n − 1); CR = CI/RI; RI = {ahp_ri()}. Matris, CR dört ondalığa "
         f"yuvarlandığında 0.10 veya altındaysa tutarlı sayılır (`round(CR, 4) ≤ 0.10`).",
         "- **Grup matrisi.** Her hücre için geometrik ortalama: g_ij = (Π_k a_ij^(k))^(1/K). Grup ağırlıkları bu "
         "matrisin özvektörüdür. Uzman grubu K = 7, bir koşulun LLM paneli K = 9 model × 5 tekrar = 45 matristir; "
         "model düzeyinde K = 5 tekrardır.",
         "- **Global ağırlıklar.** Teslimat alt ölçütlerinin yerel ağırlıkları Teslimat ana ağırlığıyla çarpılır; "
         "diğer dört ana ölçüt ağırlığı aynen alınır: (D1…D4) = w_Teslimat · (d1…d4); F, Q, V, C = ana ağırlıklar. "
         "Sekiz global ağırlığın toplamı 1'dir.",
         "- **Teslimat göstergelerinin orana çevrilmesi.** D1–D4 sipariş sayılarıdır. Her tedarikçi için "
         "D_j' = 100 · D_j / (D1 + D2 + D3 + D4). Ana tablolar (T11, T13–T15) oranlarla hesaplanır; T12 sayılarla ve "
         f"oranlarla karşılaştırır. Gösterge yönleri (`performance_matrix.json`): {ben}.",
         "- **TOPSIS.** r_ij = x_ij / √(Σ_i x_ij²); v_ij = w_j · r_ij. İdeal çözüm fayda göstergesinde en büyük, "
         "maliyet göstergesinde en küçük v_ij; ideal olmayan çözüm tersidir. D_i+ ve D_i− Öklid uzaklıklarıdır; "
         "puan C_i = D_i− / (D_i+ + D_i−).",
         "- **ARAS (η = 1).** Maliyet göstergeleri 1/x ile çevrilir. İdeal satır x_0j = max_i x_ij eklenir; "
         "normalleştirme x̄_ij = x_ij / (Σ_i x_ij + x_0j). S_i = Σ_j w_j x̄_ij, puan K_i = S_i / S_0. Kodda ek bir üs "
         "ya da katsayı yoktur.",
         "- **EDAS.** AV_j = sütun ortalaması. Fayda göstergesinde PDA = max(0, (x − AV)/AV), NDA = max(0, (AV − x)/AV); "
         "maliyet göstergesinde tersi. SP_i = Σ w_j PDA_ij, SN_i = Σ w_j NDA_ij; NSP = SP/max SP, NSN = 1 − SN/max SN; "
         "puan AS_i = (NSP + NSN)/2.",
         "- **Sıralar.** Büyük puan 1. sıradır; eşitlikte ortalama sıra verilir.",
         "- **Benzerlik.** Ağırlık vektörleri arasında Spearman ρ, Kendall τ-b, RMSE ve ortalama mutlak fark; son "
         "ikisi yüzde puan (ağırlık × 100) cinsindendir.", ""])

    # ------------------------------------------------------------ 5
    ext(["## 5. Orkestrasyon ve revizyon", "", "### 5.1 Akış", "",
         "Revizyon yalnız yönlendirmesiz – doğrudan (direct) ve CoT koşullarında, ana deneyin ilk yanıtlarından CR > 0.10 olan matrisler için "
         f"yapıldı. Hakem model: `{JUDGE['id']}` (sıcaklık {JUDGE['temperature']}, reasoning parametresi "
         "gönderilmedi). Her zincir şu adımlarla ilerler:", "",
         "1. Kod, o ana kadarki en iyi geçerli yanıtları seçer: ilk yanıt ve ayrıştırılan tüm revizyonlar içinde CR'si "
         "en düşük olan (eşitlikte önceki).",
         "2. Kod bu yanıtların CR'sini, en tutarsız üç üçlüyü (|ln a_ij + ln a_jk − ln a_ik| büyükten küçüğe) ve en çok "
         "üç döngüsel (sıralı) üçlüyü hesaplar.",
         "3. Hakem model bu listeyi, yanıtları ve CR'yi alır; en çok 120 sözcüklük bir geri bildirim yazar. "
         "Değer önermesi ve hangi ölçütün daha önemli olduğunu söylemesi yasaktır.",
         "4. Panel modeline şu konuşma gönderilir: özgün mesajlar + tek bir asistan turu (özgün yanıt ya da en iyi "
         "yanıtlar, koşulun JSON biçiminde) + yeni kullanıcı mesajı. Yeni mesaj hakemin geri bildirimi ve kodun "
         "eklediği sabit kuyruktan oluşur. Hakem geri bildirimi okunamazsa kodun yazdığı revizyon istemi kullanılır.",
         "5. Kod yanıtı ayrıştırır ve CR'yi hesaplar. Yanıt ayrıştırılabildiyse hakem, ilk yanıtlarla revize "
         "yanıtları karşılaştırıp `logic_preserved` = yes / partly / no kararı ve kısa bir yorum verir.",
         "6. Durma kuralı: `round(CR, 4) ≤ 0.10` olan ilk revizyon ya da 5 revizyon. Zincirin son yanıtları en iyi "
         "geçerli yanıtlardır (en düşük CR). Ayrıştırılamayan yanıt bir revizyon hakkı sayılır ve aynı konuşma "
         "yeniden gönderilir.", "",
         "Kodun hakem geri bildiriminin sonuna eklediği kuyruk (`{previous_answers}` yerine en iyi yanıtlar "
         "`Q1: Left … | Right … = değer` biçiminde yazılır):", ""] + fence(revise.JUDGE_FORMAT_TAIL.strip("\n")) + [
         "", "Hakem geri bildirimi kullanılamazsa gönderilen kod istemi:", ""] + fence(revise.REVISION_TEMPLATE) + [""])
    ext(["### 5.2 Kim neye karar verir", "",
         md([{"Adım": "CR, tutarsız ve döngüsel üçlüler", "Karar veren": "kod"},
             {"Adım": "Geri bildirim metni", "Karar veren": "hakem model"},
             {"Adım": "Önceki yanıtların listesi, ölçek ve biçim talimatı", "Karar veren": "kod (sabit kuyruk)"},
             {"Adım": "Revize değerler", "Karar veren": "panel modeli"},
             {"Adım": "Revize CR ve tutarlılık", "Karar veren": "kod"},
             {"Adım": "Mantık korunumu kararı (yes/partly/no)", "Karar veren": "hakem model (yalnız kayıt; durmayı etkilemez)"},
             {"Adım": "Durma ve en iyi yanıtın seçimi", "Karar veren": "kod"}]), "",
         "Kayıt: her panel çağrısı `revisions.jsonl` dosyasına (gönderilen konuşma, dayanılan yanıtlar ve CR, hakem "
         "geri bildirimi, hakem kararı, mantık ölçüleri, maliyetler), her hakem çağrısı `judge_calls.jsonl` dosyasına "
         "(amaç `feedback` ya da `verdict`, mesajlar, ham yanıt, ayrıştırma sonucu) yazılır.", ""])
    vc = rv["verdict_counts"]
    ext(["### 5.3 Revizyon denetimi ve sonuçları", "",
         f"`{rv['tag']}` etiketinde: revizyon gereken matris {rv['chains_expected']}, kaydı olan zincir "
         f"{rv['chains_present']}; panel çağrısı {rv['calls']} (başarılı {rv['calls_ok']}, ayrıştırılan "
         f"{rv['parse_ok']}); hakem çağrısı {rv['judge_calls']} "
         f"({', '.join(f'{k} {v}' for k, v in sorted(rv['judge_calls_by_purpose'].items()))}; başarılı "
         f"{rv['judge_calls_ok']}, ayrıştırılan {rv['judge_parse_ok']}). Hakem geri bildirimi okunamadığı için kod "
         f"istemine düşülen çağrı {rv['feedback_fallback_code_prompt']}, yeniden kullanılan geri bildirim "
         f"{rv['feedback_reused']}; geçerli karar {rv['verdict_valid']}, geçersiz karar {rv['verdict_invalid']}, karar "
         f"istenmeyen çağrı {rv['verdict_not_requested']}. Tüm çağrılardaki kararlar: yes {vc.get('yes', 0)}, partly "
         f"{vc.get('partly', 0)}, no {vc.get('no', 0)}. Tutarlılığa ulaşan zincir {rv['chains_reached']}. Kural "
         f"ihlali (numara sırası, durma kuralı, en iyi yanıt kuralı, konuşma yapısı, kuyruk metni, hakem çağrısı "
         f"sayısı) {len(rv['violations'])}. Hakem modeli kimliği: {', '.join(rv['judge_models'])}.", ""])
    t16 = table(tables, "T16")
    t16p = panel(t16)
    if t16p:
        need = sum(int(r["Needing revision"]) for r in t16p)
        reach = sum(int(r["Reached after ≤5"]) for r in t16p)
        ext([f"T16 panel satırları (toplam revizyon gereken {need}, 5 revizyon içinde tutarlı hale gelen {reach}, "
             f"ulaşmayan {sum(int(r['Not reached']) for r in t16p)}):", "",
             md(t16p, ["Condition", "Task", "Matrices", "Needing revision", "Reached after ≤1", "Reached after ≤2",
                       "Reached after ≤3", "Reached after ≤4", "Reached after ≤5", "Not reached", "Mean initial CR",
                       "Mean final CR"]), ""])
    t17 = table(tables, "T17")
    if t17:
        ext(["T17 ilk ve son yanıtlarla panel ana ölçüt ağırlıkları ve uzmanlarla benzerlik:", "",
             md(t17, ["Condition", "Answers", "Main matrices", "Delivery", "Financial", "Quality", "Volume", "Compliance",
                      "Spearman rho main", "RMSE main (pp)", "Spearman rho global", "RMSE global (pp)"]), ""])
    t20p = panel(table(tables, "T20"))
    if t20p:
        ext(["T20 panel satırları (son yanıtlar ilk yanıtlarla karşılaştırılır; hakem kararları son yanıtları veren "
             "revizyona aittir):", "",
             md(t20p, ["Condition", "Task", "Chains revised", "Reached CR ≤ 0.10", "Mean revisions used",
                       "Mean answers changed", "Mean Spearman rho", "Top criterion unchanged", "Ranking unchanged",
                       "Mean max abs weight change (pp)", "Judge yes", "Judge partly", "Judge no"]), ""])
    a08 = read_csv(report_dir / "appendix" / "A08_judge_feedback.csv")
    ex = [next((r for r in a08 if r["verdict"] == v), None) for v in ("yes", "no")]
    ext(["### 5.4 Örnekler (A08'den, dosya sırasındaki ilk “yes” ve ilk “no” kararı)", ""])
    for r in ex:
        if not r:
            continue
        ext([f"**{r['model']}, {r['condition']}, {r['task']}, tekrar {r['rep']}, revizyon {r['revision']}** "
             f"(CR {r['CR before']} → {r['CR after']}, karar: {r['verdict']})", "",
             f"> Geri bildirim: {r['feedback']}", ">", f"> Hakem yorumu: {r['comment']}", ""])

    # ------------------------------------------------------------ 6
    ext(["## 6. Bulgular", "",
         "Bu bölüm `results/report/report.md` tablolarını özetler. Tablolar ilk yanıtlara (`main`) dayanır; T16, "
         "T17 ve T20 `revision_final` kayıtlarını kullanır.", ""])
    _findings(ext, tables, sum(float(r.get("cost") or 0) for r in main_calls))

    # ------------------------------------------------------------ 7
    ext(["## 7. Ekler listesi", "",
         "`results/report/appendix/` klasöründeki dosyalar (satır sayısı başlık hariç):", ""])
    descr = {
        "matrices_llm.csv": "A01 Seçilen her LLM matrisinin tüm hücreleri (uzun biçim) ve matris CR'si.",
        "matrices_expert.csv": "A02 Her uzman matrisinin tüm hücreleri ve CR'si.",
        "cot_reasoning.csv": "A03 CoT koşulunda modellerin verdiği sıralama ve gerekçe metni.",
        "A04_group_matrices": "A04 Uzman ve koşul başına LLM paneli geometrik ortalama matrisleri; ağırlık, λmax, CI, CR.",
        "A05_model_matrices": "A05 Model × koşul × görev geometrik ortalama matrisleri (5 tekrar); ağırlık, λmax, CI, CR.",
        "A06_group_matrices_after_revision": "A06 Revizyon sonrası (direct, CoT) LLM paneli geometrik ortalama matrisleri.",
        "A07_run_level": "A07 Çağrı düzeyinde tablo: CR, ağırlıklar, sağlayıcı, token, maliyet, ayrıştırma bayrakları, "
                         "revizyon gereksinimi, revizyon sayısı, son CR, hakem kararları.",
        "A08_judge_feedback": "A08 Revision_final hakem geri bildirimleri ve kararları (her revizyon çağrısı).",
        "README.md": "Bu klasördeki ve tables/ klasöründeki dosyaların İngilizce listesi (açıklama ve satır sayısı).",
    }
    rows = []
    for p in sorted((report_dir / "appendix").iterdir()):
        n = len(read_csv(p)) if p.suffix == ".csv" else None
        d = descr.get(p.name) or descr.get(p.stem, "")
        if p.suffix == ".tex":
            d = f"{p.stem}.csv dosyasının LaTeX (longtable) karşılığı."
        rows.append({"Dosya": f"appendix/{p.name}", "Satır": "" if n is None else n, "Açıklama": d})
    ext([md(rows), "",
         "Denetim çıktıları `results/report/audit/` altındadır (`audit.md` ve a–k bölümlerinin CSV dosyaları).", ""])

    # ------------------------------------------------------------ 8
    t03p = {r["Condition"]: r for r in panel(table(tables, "T03"))}
    mm = settings.get("minimax-m2.5", {})
    judge_ids = {}
    for r in revs:
        judge_ids.setdefault(r.get("run_tag"), set()).add(str(r.get("judge_model_id")))
    judge_ids = {k: sorted(v) for k, v in judge_ids.items()}
    pilots_rev = [(t, inv[("revisions.jsonl", t)]["records"], inv.get(("judge_calls.jsonl", t), {}).get("records", 0))
                  for t in ("revision_judge", "revision_v2", "rev_final", "rev_test_none") if ("revisions.jsonl", t) in inv]
    lim = ["## 8. Sınırlılıklar", ""]
    if gpt0:
        lim.append(f"- **gpt-5.2 sıcaklığı.** Model sıcaklığı uygulamıyor: T = 0 ile {gpt0['calls']} çağrıda "
                   f"{gpt0['distinct_answer_sets']} farklı yanıt kümesi. Bu modelin sıcaklık kollarındaki ve ana "
                   "deneydeki değişkenliği istenen sıcaklığa bağlanamaz.")
    lim.append(f"- **minimax-m2.5 reasoning.** Reasoning kapatılamadı; ana deneyde ortalama "
               f"{mm.get('mean_reasoning_tokens', 0):.1f} reasoning token üretildi. Diğer modellerde reasoning "
               "kapalıydı (gpt-4o için parametre gönderilmedi).")
    lim.append(f"- **Sağlayıcı yönlendirmesi sabitlenmedi.** OpenRouter aynı modeli farklı sağlayıcılara yönlendirdi; "
               f"ana deneyde {len(multi)} model birden fazla sağlayıcıdan yanıt aldı (3.2-e). Sağlayıcılar arası "
               "farklar ayrıca incelenmedi.")
    lim.append(f"- **Tekrar sayısı.** Her model × koşul × görev hücresinde {audit.MAIN_REPS}, sıcaklık kollarında "
               f"{audit.TEMP_REPS} tekrar vardır. Model düzeyindeki ortalama ve dağılım ölçüleri bu az sayıda gözleme "
               "dayanır.")
    lim.append("- **Dil.** Yönlendirmesiz – doğrudan (direct) ve CoT istemleri İngilizcedir; uzmanlara sorular Türkçe sorulmuştur. Guided isteminde "
               "talimatlar İngilizce; ana ölçüt adları, açıklamaları ve KPI raporu Türkçedir.")
    lim.append("- **Olumsuz alt ölçütlerde “önem” belirsizliği.** Sorular “hangisi daha önemli” diye sorar. "
               f"Performans matrisinde {', '.join(c for c, b in zip(perf['codes'], perf['benefit_mask']) if not b)} "
               "maliyet yönlüdür (D2–D4 bekleyen, iptal edilen ve kısmen teslim edilen siparişlerdir). Soru metni, "
               "önemin istenen ya da istenmeyen bir durumun izlenmesi anlamında mı sorulduğunu belirtmez.")
    if "guided" in t03p:
        bad = [int(b) - int(a) for a, b in (t03p["guided"][f"CR ≤ 0.10 {t}"].split("/") for t in ("main", "delivery"))]
        lim.append(f"- **Revizyon kapsamı.** Revizyon yalnız yönlendirmesiz – doğrudan (direct) ve CoT koşullarında yapıldı. Guided koşulunda CR > 0.10 "
                   f"olan matris sayısı: ana görevde {bad[0]}, teslimat görevinde {bad[1]}. Bu matrisler revize edilmedi.")
    lim.append("- **Hakem bir LLM'dir.** Mantık korunumu kararları tek bir modelden gelir; dosyalarda bu kararların "
               "insan değerlendirmesiyle karşılaştırılması yoktur.")
    lim.append(f"- **Pilotlar dışarıda.** `pilot` ({inv[('calls.jsonl', 'pilot')]['records'] if ('calls.jsonl', 'pilot') in inv else 0} "
               "çağrı) ve revizyon pilotları sonuçlara alınmadı: "
               + "; ".join(f"`{t}` {a} panel + {b} hakem çağrısı" for t, a, b in pilots_rev)
               + ". Hakem modeli: " + "; ".join(f"`{t}` {', '.join(judge_ids.get(t, []))}" for t, _, _ in pilots_rev)
               + ". `revision_judge` ilk protokoldür; diğerleri kısa denemelerdir.")
    ext(lim + [""])

    # ------------------------------------------------------------ 9
    def cost(f, t):
        return inv[(f, t)]["cost_usd"] if (f, t) in inv else 0.0
    rows = [("Parametre sınamaları", cost("param_tests.jsonl", "(param test)")),
            ("Ön deneme (pilot)", cost("calls.jsonl", "pilot")),
            ("Ana deney (main)", cost("calls.jsonl", "main"))]
    rows += [(f"Sıcaklık kolu {t}", cost("calls.jsonl", t)) for t in res["temp_tags"]]
    rows += [(f"Revizyon pilotu {t} (panel + hakem)", cost("revisions.jsonl", t) + cost("judge_calls.jsonl", t))
             for t, _, _ in pilots_rev]
    rows += [("Revizyon revision_final, panel", cost("revisions.jsonl", rv["tag"])),
             ("Revizyon revision_final, hakem", cost("judge_calls.jsonl", rv["tag"]))]
    total = sum(c for _, c in rows)
    ext(["## 9. Harcama özeti", "",
         "Maliyetler OpenRouter yanıtındaki `usage.cost` alanından toplanmıştır (USD). Yedek dosyalar ana kayıtların "
         "kopyası olduğu için toplama alınmadı.", "",
         md([{"Aşama": a, "Maliyet (USD)": usd(c)} for a, c in rows] + [{"Aşama": "**Toplam**", "Maliyet (USD)": usd(total)}]),
         ""])
    return "\n".join(L)


def ahp_ri() -> str:
    from .ahp import RI
    # The study used matrices of size 4 and 5; the RI table extended to n = 9 for other problems is not listed.
    return ", ".join(f"n = {k}: {v}" for k, v in sorted(RI.items()) if k <= 5)


# ---------------------------------------------------------------- findings (section 6)

def _top(row: dict, cols) -> str:
    return max(cols, key=lambda c: float(row[c]) if row.get(c) not in ("", None) else -1)


def _findings(ext, tables: dict, main_cost: float) -> None:
    def sec(tid, title):
        ext([f"### {tid} {title}", ""])

    t = tables.get("T01")
    if t:
        sec("T01", "Modeller ve ayarlar")
        ext([f"Model başına çağrı sayısı, sağlayıcılar, uygulanan reasoning ve maliyet. Toplam çağrı "
             f"{sum(int(r['Calls']) for r in t)}, ayrıştırma hatası {sum(int(r['Parse failures']) for r in t)}, toplam "
             f"maliyet {usd(main_cost)} USD (ayrıntı 2.1).", ""])
    t = tables.get("T02")
    if t:
        sec("T02", "Uzman tutarlılığı")
        grp = next(r for r in t if r["Expert"].startswith("Group"))
        cnt = next(r for r in t if r["Expert"].startswith("n with"))
        ext([f"{len(t) - 2} uzmanın bireysel CR değerleri. CR ≤ 0.10 olan uzman matrisi: ana ölçütlerde {cnt['CR main']}, "
             f"teslimat alt ölçütlerinde {cnt['CR delivery']}. Geometrik ortalama grup matrisinin CR'si ana "
             f"ölçütlerde {grp['CR main']}, teslimatta {grp['CR delivery']}. Bireysel CR aralığı ana ölçütlerde "
             f"{min(float(r['CR main']) for r in t[:-2]):.4f}–{max(float(r['CR main']) for r in t[:-2]):.4f}, "
             f"teslimatta {min(float(r['CR delivery']) for r in t[:-2]):.4f}–{max(float(r['CR delivery']) for r in t[:-2]):.4f}.", ""])
    t = tables.get("T03")
    if t:
        sec("T03", "LLM tutarlılığı")
        p = panel(t)
        ext(["Model ve koşul başına ortalama ve en büyük CR ile CR ≤ 0.10 olan matris sayısı; panel satırları tüm "
             "modelleri birleştirir ve grup matrisinin CR'sini verir. "
             + " ".join(f"{r['Condition']}: ana görevde {r['CR ≤ 0.10 main']}, teslimatta {r['CR ≤ 0.10 delivery']} "
                        f"matris tutarlı; grup CR {r['Group CR main']} / {r['Group CR delivery']}." for r in p), "",
             md(t, ["Model", "Condition", "Mean CR main", "CR ≤ 0.10 main", "Group CR main", "Mean CR delivery",
                    "CR ≤ 0.10 delivery", "Group CR delivery"]), ""])
    for tid, title, crit in (("T04", "Ana ölçüt ağırlıkları", "Criterion"), ("T05", "Teslimat yerel ağırlıkları", "Criterion")):
        t = tables.get(tid)
        if not t:
            continue
        sec(tid, title)
        srcs = ["Experts"] + [f"LLM {cond_col(c)}" for c in an.CONDITIONS]
        tops = {s: max(t, key=lambda r: float(r[s]))[crit] for s in srcs}
        diffs = {cond_col(c): max(t, key=lambda r: abs(float(r[f"Diff {cond_col(c)} (pp)"]))) for c in an.CONDITIONS}
        ext(["Uzman grubu ve her koşulun LLM paneli için grup ağırlıkları ile LLM − uzman farkı (yüzde puan). "
             "En yüksek ağırlıklı ölçüt: " + "; ".join(f"{s}: {tops[s]}" for s in srcs) + ". En büyük mutlak fark: "
             + "; ".join(f"{c}: {r[crit]} ({r[f'Diff {c} (pp)']} pp)" for c, r in diffs.items()) + ".", "", md(t), ""])
    t = tables.get("T06")
    if t:
        sec("T06", "Global ağırlıklar")
        diffs = {cond_col(c): max(t, key=lambda r: abs(float(r[f"Diff {cond_col(c)} (pp)"]))) for c in an.CONDITIONS}
        ext(["Sekiz global ağırlık (D1–D4, F, Q, V, C) ve LLM − uzman farkı. En büyük mutlak fark: "
             + "; ".join(f"{c}: {r['Criterion']} ({r[f'Diff {c} (pp)']} pp)" for c, r in diffs.items()) + ".", ""])
    t = tables.get("T07")
    if t:
        sec("T07", "Model düzeyinde ana ağırlıklar")
        txt = []
        for c in an.CONDITIONS:
            cnt = Counter(r["Top criterion"] for r in t if r["Condition"] == cond_label(c))
            txt.append(f"{cond_label(c)}: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items(), key=lambda x: (-x[1], x[0]))))
        ext(["Her modelin 5 tekrarının geometrik ortalamasından ana ölçüt ağırlıkları ve en yüksek ağırlıklı ölçüt. "
             "En yüksek ölçüte göre model sayısı: " + "; ".join(txt) + ".", ""])
    t = tables.get("T08")
    if t:
        sec("T08", "Uzman ağırlıklarına benzerlik")
        ext(["Model ve panel ağırlıklarının uzman ağırlıklarına Spearman ρ, Kendall τ, RMSE ve ortalama mutlak fark "
             "(pp) değerleri; ana (5) ve global (8) ağırlıklar için. Panel satırları:", "",
             md(panel(t), ["Condition", "Spearman rho main", "Kendall tau main", "RMSE (pp) main",
                           "Mean abs diff (pp) main", "Spearman rho global", "Kendall tau global", "RMSE (pp) global",
                           "Mean abs diff (pp) global"]), ""])
    t = tables.get("T09")
    if t:
        sec("T09", "Tekrarlar arası kararlılık")
        txt = []
        for c in an.CONDITIONS:
            rs = [r for r in t if r["Condition"] == cond_label(c) and r["Mean CV"]]
            n_multi = sum(1 for r in rs if r["Distinct top criteria"] and int(r["Distinct top criteria"]) > 1)
            txt.append(f"{cond_label(c)}: ortalama CV {np.mean([float(r['Mean CV']) for r in rs]):.3f}, birden fazla farklı en "
                       f"yüksek ölçüt veren model {n_multi}/{len(rs)}")
        ext(["Model × koşul başına ana ağırlıkların tekrarlar arası standart sapması ve değişim katsayısı (CV) ile "
             "tekrarlarda görülen farklı en yüksek ölçüt sayısı. " + "; ".join(txt) + ".", ""])
    t = tables.get("T10")
    if t:
        sec("T10", "Panel bileşimi")
        ext(["Tüm modeller, geliştirici başına bir model (claude-haiku-4.5 ve gpt-5.2 çıkarılmış) ve gpt-5.2'siz "
             "panel için ana ağırlıklar ve uzmanlara benzerlik:", "",
             md(t, ["Condition", "Panel", "Models", "Spearman rho main", "RMSE (pp) main", "Spearman rho global",
                    "RMSE (pp) global"]), ""])
    t = tables.get("T11")
    if t:
        sec("T11", "Tedarikçi sıralamaları (oranlar)")
        rows = []
        for src in dict.fromkeys(r["Weights"] for r in t):
            rs = [r for r in t if r["Weights"] == src]
            rows.append({"Ağırlıklar": src, **{f"{mth} 1. sıra": next(r["Supplier"] for r in rs if float(r[f"{mth} rank"]) == 1)
                                               for mth in an.METHODS}})
        ext(["Uzman ve panel global ağırlıklarıyla TOPSIS, ARAS ve EDAS puanları ve sıraları (teslimat göstergeleri "
             "oran). Birinci sıradaki tedarikçi:", "", md(rows), "",
             md([{"Ağırlıklar": r["Weights"], "Tedarikçi": r["Supplier"], "TOPSIS": r["TOPSIS rank"],
                  "ARAS": r["ARAS rank"], "EDAS": r["EDAS rank"]} for r in t]), ""])
    t = tables.get("T12")
    if t:
        sec("T12", "Sayılar ve oranlar")
        n_diff = sum(1 for r in t for mth in an.METHODS if r[f"{mth} rank counts"] != r[f"{mth} rank rates"])
        ext([f"Uzman ve yönlendirmesiz – doğrudan (direct) panel ağırlıklarıyla, teslimat göstergeleri sayı ve oran olarak alındığında sıralar. "
             f"{len(t)} satır × 3 yöntemde sayı ve oran sıraları {n_diff} hücrede farklıdır.", ""])
    t = tables.get("T13")
    if t:
        sec("T13", "Eşit ağırlık karşılaştırması")
        ext(["Sekiz global ağırlık 1/8 iken sıralar. Birinci sıra: "
             + "; ".join(f"{mth}: {next(r['Supplier'] for r in t if float(r[f'{mth} rank']) == 1)}" for mth in an.METHODS)
             + ".", ""])
    t = tables.get("T14")
    if t:
        sec("T14", "Ana ölçüt ağırlığı duyarlılığı")
        lv = sorted({r["Level"] for r in t if r["Level"]})
        ext([f"Her ana ölçütün ağırlığı {lv[0]}–{lv[-1]} arasında 0.05 adımla değiştirilir, diğer dört ana ölçüt "
             f"kalanı eşit paylaşır; Teslimat ağırlığı D1–D4'e eşit bölünür. Tablo {len(t)} satırdır (ilk satır eşit "
             "ağırlık başvurusu) ve her yöntem için sıralamayı ve birinci tedarikçiyi verir.", ""])
    t = tables.get("T15")
    if t:
        sec("T15", "Duyarlılık özeti")
        never = [f"{r['Varied criterion']}/{r['Method']}" for r in t if not r["Top supplier first differs at"]]
        ext(["T14'ün ana ölçüt × yöntem özeti. Birinci tedarikçinin hiç değişmediği durumlar: "
             + (", ".join(never) if never else "yok") + ".", "", md(t), ""])
    t = tables.get("T16")
    if t:
        sec("T16", "Revizyon özeti")
        ext(["Model, koşul ve görev başına revizyon gereken matris sayısı ve 1–5 revizyon içinde CR ≤ 0.10'a ulaşan "
             "birikimli sayı, ortalama ilk ve son CR. Panel satırları 5.3'tedir.", ""])
    t = tables.get("T17")
    if t:
        sec("T17", "Revizyon öncesi ve sonrası ağırlıklar")
        ext(["İlk ve son yanıtlarla panel ağırlıkları ve uzmanlara benzerlik (tablo 5.3'te).", ""])
    t = tables.get("T18")
    if t:
        sec("T18", f"Sıcaklık karşılaştırması ({DIRECT})")
        ext(["Yönlendirmesiz – doğrudan (direct) koşulda sıcaklık 0.0, 0.6 ve 1.0 için panel ana ağırlıkları, tutarlı matris sayısı ve tekrarlar "
             "arası ortalama CV. Sıcaklık kollarında 3, ana deneyde 5 tekrar vardır.", "", md(t), ""])
    t = tables.get("T20")
    if t:
        sec("T20", "Revizyon sonrası mantık korunumu")
        p = panel(t)
        ext(["Revize edilen zincirlerde son yanıtların ilk yanıtlarla karşılaştırılması ve hakem kararları. Panel "
             f"toplamı: yes {sum(int(r['Judge yes']) for r in p)}, partly {sum(int(r['Judge partly']) for r in p)}, "
             f"no {sum(int(r['Judge no']) for r in p)} (son yanıtları veren revizyon başına; panel satırları 5.3'te).", ""])
    t = tables.get("T21")
    if t:
        sec("T21", "Uzman ağırlıklı sıralamayla uyum")
        ext(["TOPSIS, ARAS ve EDAS ile (teslimat göstergeleri oran) her koşulun LLM paneli ağırlıklarıyla ve eşit "
             "ağırlıklarla elde edilen tedarikçi sıralarının uzman ağırlıklarıyla elde edilen sıralarla Spearman ρ ve "
             "Kendall τ değerleri ile sırası farklı olan tedarikçi sayısı (sıralar T11 ve T13'teki gibidir).", "",
             md(t), ""])
    t = tables.get("T22")
    if t:
        sec("T22", "Yanıtlar ve yönlendirmeli istemdeki sıra")
        g = [r for r in t if r["Comparison"] == "prompted" and r["Respondents"] == "Panel"]
        ext(["Yönlendirmeli istemde yazan sıranın yanıtlarda izlenip izlenmediği (sıra istem dosyasından okunur: ana "
             "ölçütlerde `priority_guidance` içindeki numaralı sıra; teslimatta yalnız birinci ölçüt, “PRIMARY positive "
             "indicator” olarak anılan teslim edilen siparişler). Panel (guided): "
             + "; ".join(f"{r['Task']}: {r['Replies']} yanıtın {r['Ranking equals prompted order'] or '–'} tanesinde "
                         f"sıralama istemdeki sırayla aynı, {r['Prompted first criterion ranked first']} tanesinde "
                         f"istemdeki birinci ölçüt birinci" for r in g)
             + ". Doğrudan ve CoT yanıtları ile uzmanların bireysel matrisleri bu sırayı görmemiştir ve yalnız "
             "karşılaştırma (reference) içindir.", "", md(t), ""])
    ext(["### Şekiller", "",
         "- F1 (`figures/F1_global_weight_diff_heatmap.png`): model − uzman global ağırlık farkı (pp), koşul başına bir panel.",
         "- F2 (`figures/F2_cr_main_dotplot.png`): ana ölçüt matrislerinin CR değerleri, model ve koşul başına; kesikli "
         "çizgi CR = 0.10.", ""])


def main(argv=None) -> int:
    from . import cli
    ap = argparse.ArgumentParser(prog="python -m ahp_bench.tech_report")
    ap.add_argument("--results", default=None, help=f"results folder of the study ({cli.STUDY_REPO_URL})")
    cli.add_data(ap)
    ap.add_argument("--report", default=None, help="report folder with tables/ and appendix/ (default: <results>/report)")
    a = ap.parse_args(argv)
    from .report import write_text
    results, data = cli.study_paths(a)
    report_dir = cli.resolve(a.report) if a.report else results / "report"
    text = build(results, data, report_dir)
    out = report_dir / "TEKNIK_RAPOR.md"
    write_text(out, text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
