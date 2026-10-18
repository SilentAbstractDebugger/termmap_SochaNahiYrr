"""Candidate generation (Stage D merge + Stage E hierarchy expansion) and code-agnostic features.

No feature encodes code identity. Every feature is a property of (query, candidate description,
candidate position in the hierarchy, training support), so codes never seen in training compete
on equal terms with seen ones.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .context import build_context
from .normalize import STOPWORDS, QueryVariant, Vocab, basic_norm, make_variants
from .qualifiers import compare, extract, token_supported
from .retrieve_lexical import LexicalIndex
from .retrieve_training import TrainingResources
from .terminology import KB, icd_nodot
from .util import stable_rng

TOP_TAGS = ("disorder", "finding", "morphologic abnormality", "situation", "observable entity",
            "procedure", "body structure", "qualifier value", "substance", "event")
UNSPEC_WORDS = ("unspecified", "nos")
KINDS = ("orig", "rep", "abbr", "syn")
SCORE_KEYS = tuple(f"{p}_{k}" for p in ("exact", "exact_sorted", "bm25", "char") for k in KINDS) + (
    "abbr_conf", "abbr_ctx_support", "nb_max", "nb_count", "nb_sum", "nb_ctx_max", "nb_x_ctx", "alias_exact", "hier_added")
RANK_KEYS = tuple(f"{p}_rank_{k}" for p in ("bm25", "char") for k in KINDS)


def build_vocab(kbs: dict[str, KB], cfg: dict) -> Vocab:
    c: Counter = Counter()
    for kb in kbs.values():
        for t in kb.doc_text:
            c.update(t.split())
    n = cfg["normalize"]
    return Vocab(c, n["typo_min_len"], n["typo_max_distance"], n["prefix_min_len"])


def merge_abbr(kbs: dict[str, KB], tres: list[TrainingResources]) -> dict[str, list[tuple[str, float]]]:
    """Training-mined expansions first (conf 1.0), then dictionary-mined (conf 0.8)."""
    tr: dict[str, Counter] = defaultdict(Counter)
    for r in tres:
        for a, cnt in r.abbr.items():
            tr[a].update(cnt)
    dc: dict[str, Counter] = defaultdict(Counter)
    for kb in kbs.values():
        for a, cnt in kb.abbr.items():
            dc[a].update(cnt)
    trans: dict[str, Counter] = defaultdict(Counter)
    for r in tres:
        for a, cnt in getattr(r, "trans", {}).items():
            trans[a].update(cnt)
    out: dict[str, list[tuple[str, float]]] = {}
    for a in sorted(set(tr) | set(dc) | set(trans)):
        lst = [(e, 1.0) for e, _ in sorted(tr[a].items(), key=lambda kv: (-kv[1], kv[0]))]
        seen = {e for e, _ in lst}
        lst += [(e, 0.8) for e, _ in sorted(dc[a].items(), key=lambda kv: (-kv[1], kv[0])) if e not in seen]
        seen = {e for e, _ in lst}
        lst += [(e, 0.6) for e, _ in sorted(trans[a].items(), key=lambda kv: (-kv[1], kv[0]))[:2] if e not in seen]
        out[a] = lst
    return out


class FeatureBuilder:
    def __init__(self, cfg: dict, kbs: dict[str, KB], lex: dict[str, LexicalIndex], vocab: Vocab,
                 dense=None):
        self.cfg, self.kbs, self.lex, self.vocab = cfg, kbs, lex, vocab
        # dense: None | one index (legacy, feature prefix "dense_") | {name: index} ("dense_<name>_")
        if dense is None or isinstance(dense, dict):
            self.dense = dense or None
        else:
            self.dense = {"": dense}
        self.ce = None  # optional cross-encoder scorer (set by pipeline when configured)
        self._bm25_cache: dict[tuple[str, str], list] = {}
        self._char_cache: dict[tuple[str, str], list] = {}
        self._dense_cache: dict[tuple[str, str], list] = {}

    # ------------------------------------------------------------------ retrieval helpers
    def _batch_search(self, system: str, texts: list[str]) -> None:
        r = self.cfg["retrieval"]
        lex = self.lex[system]
        todo = sorted({t for t in texts if (system, t) not in self._bm25_cache})
        if todo:
            for t, res in zip(todo, lex.search_bm25(todo, r["top_k_bm25"])):
                self._bm25_cache[(system, t)] = res
            for t, res in zip(todo, lex.search_char(todo, r["top_k_char"])):
                self._char_cache[(system, t)] = res
        for name, idx in (self.dense or {}).items():
            todo = sorted({t for t in texts if (name, system, t) not in self._dense_cache})
            if todo:
                k = int(getattr(idx, "top_k", None) or self.cfg["dense"]["top_k"])
                for t, res in zip(todo, idx.search(system, todo, k)):
                    self._dense_cache[(name, system, t)] = res

    def clear_caches(self) -> None:
        """Bound memory between folds (retrieval results are re-computable and deterministic)."""
        import gc

        self._bm25_cache.clear()
        self._char_cache.clear()
        self._dense_cache.clear()
        gc.collect()

    def _dense_hits(self, system, lex, text, kind, hit, only_context_view=False):
        for name, idx in (self.dense or {}).items():
            if only_context_view and not getattr(idx, "context_view", False):
                continue
            pre = "dense" if name == "" else f"dense_{name}"
            dc = getattr(idx, "doc_codes", None)
            dc = dc[system] if isinstance(dc, dict) else lex.doc_codes
            for r, (di, sc) in enumerate(self._dense_cache.get((name, system, text), [])):
                for c in dc[di]:
                    hit(c, f"{pre}_{kind}", sc, f"{pre}_rank_{kind}", r)

    # ------------------------------------------------------------------ main entry
    def build(self, queries: pd.DataFrame, system: str, tres: TrainingResources,
              abbr: dict, gold: dict | None = None, dropout: float = 0.0,
              seed: int = 0) -> pd.DataFrame:
        """queries: qid, entity, chunk_text. gold: qid -> gold code (for labels / dropout)."""
        kb, lex = self.kbs[system], self.lex[system]
        ncfg, rcfg = self.cfg["normalize"], self.cfg["retrieval"]
        win_t = self.cfg["context"]["window_tokens"]
        qs = list(queries.itertuples(index=False))
        variants = {q.qid: make_variants(q.entity, self.vocab, abbr, ncfg["max_abbr_expansions"],
                                         ncfg["max_variants"]) for q in qs}
        # Cross-dictionary synonym bridge (thesaurus use of dictionary TEXT, not a code crosswalk):
        # strong lexical hits in the OTHER terminology contribute their synonym strings as extra
        # retrieval views for this terminology.
        other = "SNOMEDCT_US" if system == "ICD10CM" else "ICD10CM"
        if rcfg.get("bridge_enabled", True) and other in self.lex:
            okb, olex = self.kbs[other], self.lex[other]
            self._batch_search(other, [v.text for vs in variants.values() for v in vs[:2]])
            for q in qs:
                vs = variants[q.qid]
                seen = {v.text for v in vs}
                hits: dict[str, float] = {}
                for v in vs[:2]:
                    for c in olex.exact(v.text):
                        hits[c] = max(hits.get(c, 0.0), 1.0)
                    for di, sc in self._char_cache[(other, v.text)][:3]:
                        if sc >= rcfg.get("bridge_min_sim", 0.7):
                            for c in olex.doc_codes[di][:3]:
                                hits[c] = max(hits.get(c, 0.0), sc)
                added = 0
                for c, sc in sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))[:3]:
                    for d in okb.descs_norm[c][:6]:
                        if d not in seen and added < rcfg.get("bridge_max_variants", 10):
                            seen.add(d)
                            vs.append(QueryVariant(d, "syn", sc, {"bridge": other}))
                            added += 1
        self._batch_search(system, [v.text for vs in variants.values() for v in vs])
        if self.dense and any(getattr(i, "context_view", False) for i in self.dense.values()):
            ctx_texts = []
            for q in qs:
                e = basic_norm(q.entity)
                w = build_context(q.entity, q.chunk_text, win_t).window
                if w and (e + " ; " + w).strip(" ;") != e:
                    ctx_texts.append((e + " ; " + w).strip(" ;"))
            for name, idx in self.dense.items():
                if getattr(idx, "context_view", False):
                    todo = sorted({t for t in ctx_texts if (name, system, t) not in self._dense_cache})
                    k = int(getattr(idx, "top_k", None) or self.cfg["dense"]["top_k"])
                    for t, res in zip(todo, idx.search(system, todo, k) if todo else []):
                        self._dense_cache[(name, system, t)] = res
        # training neighbours: original + first repaired view
        nb_texts, nb_owner = [], []
        for q in qs:
            vs = variants[q.qid]
            for v in vs[:2]:
                nb_texts.append(v.text)
                nb_owner.append(q.qid)
        nb_res = tres.neighbours(nb_texts, rcfg["top_k_train"])
        nb_by_q: dict = defaultdict(list)
        for owner, res in zip(nb_owner, nb_res):
            nb_by_q[owner].extend(res)

        rows, parts = [], []

        def flush():
            if rows:
                part = pd.DataFrame(rows)
                num = [c for c in part.columns if c not in ("qid", "code")]
                part[num] = part[num].astype(np.float32)  # memory: float32 feature tables
                parts.append(part)
                rows.clear()

        for q in qs:
            g = gold.get(q.qid) if gold else None
            excl = None
            if g and dropout > 0 and stable_rng(seed, "dropout", system, q.qid).random() < dropout:
                excl = g
            rows.extend(self._query_rows(q, system, kb, lex, tres, abbr, variants[q.qid],
                                         nb_by_q[q.qid], excl, g, win_t, rcfg))
            if len(rows) >= 20000:
                flush()
        flush()
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if len(df):
            df = df.sort_values(["qid", "code"], kind="mergesort").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ per query
    def _query_rows(self, q, system, kb: KB, lex: LexicalIndex, tres: TrainingResources, abbr,
                    vs: list[QueryVariant], nbs, excl, gold, win_t, rcfg) -> list[dict]:
        ent = basic_norm(q.entity)
        ctx = build_context(q.entity, q.chunk_text, win_t)
        ent_has_abbr = float(any(t in abbr and len(t) <= 5 for t in ent.split()))
        cand: dict[str, dict] = defaultdict(lambda: defaultdict(float))
        ranks: dict[str, dict] = defaultdict(dict)

        def hit(code, key, val, rank_key=None, rank=None):
            c = cand[code]
            if val > c[key]:
                c[key] = val
            if rank_key is not None:
                old = ranks[code].get(rank_key)
                if old is None or rank < old:
                    ranks[code][rank_key] = rank

        chunk_set = set(ctx.clean.split())
        for vi, v in enumerate(vs):
            kind = ("orig" if v.kind == "original" else "abbr" if v.kind.startswith("abbr")
                    else "syn" if v.kind == "syn" else "rep")
            for c in lex.exact(v.text):
                hit(c, f"exact_{kind}", 1.0)
            for c in lex.exact_sorted(v.text):
                hit(c, f"exact_sorted_{kind}", 1.0)
            for r, (di, sc) in enumerate(self._bm25_cache[(system, v.text)]):
                for c in lex.doc_codes[di]:
                    hit(c, f"bm25_{kind}", sc, f"bm25_rank_{kind}", r)
            for r, (di, sc) in enumerate(self._char_cache[(system, v.text)]):
                for c in lex.doc_codes[di]:
                    hit(c, f"char_{kind}", sc * (v.weight if kind != "orig" else 1.0), f"char_rank_{kind}", r)
                    if kind == "abbr":
                        hit(c, "abbr_conf", v.weight)
                        # does the chunk context support THIS expansion? (abbreviation collision)
                        exp = [t for t in v.meta.get("expansion", "").split() if t not in STOPWORDS]
                        if exp:
                            hit(c, "abbr_ctx_support", sum(t in chunk_set for t in exp) / len(exp))
            self._dense_hits(system, lex, v.text, kind, hit)
        if self.dense:
            ctx_q = (ent + " ; " + ctx.window).strip(" ;") if ctx.window else ""
            if ctx_q and ctx_q != ent:
                self._dense_hits(system, lex, ctx_q, "ctx", hit, only_context_view=True)

        # training neighbours (code dropout: neighbours/alias carrying the excluded code vanish)
        nb_codes: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for i, s in nbs:
            c = tres.codes[i]
            if c == excl:
                continue
            nb_codes[c].append((i, s))
        nb_idx = sorted({i for lst in nb_codes.values() for i, _ in lst})
        nb_ctx = dict(zip(nb_idx, tres.ctx_sims(ctx.window, nb_idx)))
        for c, lst in nb_codes.items():
            hit(c, "nb_ctx_max", max(nb_ctx.get(i, 0.0) for i, _ in lst))
            # neighbour score weighted by context agreement (local coder convention + context)
            hit(c, "nb_x_ctx", max(s * (0.5 + nb_ctx.get(i, 0.0)) for i, s in lst))
            best = max(s for _, s in lst)
            hit(c, "nb_max", best)
            hit(c, "nb_count", float(len({i for i, _ in lst})))
            hit(c, "nb_sum", float(sum(s for _, s in {i: s for i, s in lst}.items())))
        if ent in tres.alias and tres.alias[ent] != excl:
            hit(tres.alias[ent], "alias_exact", 1.0)

        # first-stage score for pruning / hierarchy mass
        def s0(f):
            def mx(p):
                return max(f.get(f"{p}_{k}", 0.0) for k in KINDS)
            dmax = max([v for k, v in f.items() if k.startswith("dense") and "_rank_" not in k], default=0.0)
            return (mx("char") + 0.05 * mx("bm25") + 0.5 * mx("exact") + 0.6 * f.get("nb_max", 0)
                    + 1.0 * f.get("alias_exact", 0) + 0.5 * dmax)

        scored = sorted(((s0(f), c) for c, f in cand.items() if kb.valid(c)), key=lambda x: (-x[0], x[1]))
        scored = scored[: rcfg["max_candidates"]]
        keep = {c for _, c in scored}
        s0map = {c: s for s, c in scored}

        # ICD hierarchy expansion (Stage E): ancestors + unspecified/other siblings + children
        if system == "ICD10CM":
            for _, c in scored[: rcfg["hier_expand_top"]]:
                extra = list(kb.ancestors(c))
                par = kb.parent.get(c)
                if par:
                    extra += kb.children.get(par, [])
                extra += kb.children.get(c, [])
                for e in extra:
                    if e not in keep and kb.valid(e):
                        keep.add(e)
                        s0map[e] = 0.0
                        cand[e]["hier_added"] = 1.0
        codes = sorted(keep)
        if not codes:
            return []

        # ---------- string / context / qualifier features for every candidate
        vtexts = [v.text for v in vs]
        vtok_union = set()
        for v in vs:
            vtok_union.update(v.text.split())
        # tokens literally present in the term (original + repaired views, NOT abbreviation
        # expansions / synonyms) -- context support must be measured against these
        lit_toks = set()
        for v in vs:
            if v.kind in ("original",) or not (v.kind.startswith("abbr") or v.kind == "syn"):
                lit_toks.update(v.text.split())
        etoks = [t for t in ent.split() if t not in STOPWORDS]
        rel = tres.relevance
        ew = [rel.weight(t) for t in etoks]
        ew_sum = sum(ew) or 1.0
        wtoks = set(ctx.window.split()) - set(STOPWORDS)
        qual_e = extract(ent)
        qual_w = extract(ctx.window)
        chunk_toks = set(ctx.clean.split())
        # hierarchy mass
        cat_mass: Counter = Counter()
        tot_mass = 0.0
        for c in codes:
            cat_mass[kb.category(c)] += s0map.get(c, 0.0)
            tot_mass += s0map.get(c, 0.0)
        top_code = scored[0][1] if scored else None
        top_anc = set(kb.ancestors(top_code)) if (top_code and system == "ICD10CM") else set()
        cat_rank = {cat: i for i, (cat, _) in enumerate(sorted(cat_mass.items(), key=lambda kv: (-kv[1], kv[0])))}
        s0_rank = {c: i for i, (_, c) in enumerate(sorted(((s0map.get(c, 0.0), c) for c in codes),
                                                          key=lambda x: (-x[0], x[1])))}
        n_codes = len(codes)
        e_char = lex.char_vec([ent or "_"])
        out = []
        for c in codes:
            f = cand[c]
            dn = kb.descs_norm[c][:10]
            # best description by token_set_ratio against any variant
            best_d, best_sc, best_v = dn[0], -1.0, vtexts[0] if vtexts else ent
            for d in dn:
                for vt in vtexts:
                    sc = fuzz.token_set_ratio(vt, d)
                    if sc > best_sc or (sc == best_sc and len(d) < len(best_d)):
                        best_d, best_sc, best_v = d, sc, vt
            row = {"qid": q.qid, "code": c}
            for k in SCORE_KEYS:
                row[k] = f.get(k, 0.0)
            for k in RANK_KEYS:
                row[k] = ranks[c].get(k, 99)
            for name, idx in (self.dense or {}).items():
                pre = "dense" if name == "" else f"dense_{name}"
                kinds = KINDS + (("ctx",) if getattr(idx, "context_view", False) else ())
                for kd in kinds:
                    row[f"{pre}_{kd}"] = f.get(f"{pre}_{kd}", 0.0)
                    row[f"{pre}_rank_{kd}"] = ranks[c].get(f"{pre}_rank_{kd}", 99)
            rr = [v for k, v in ranks[c].items() if v < 99]
            row["rrf"] = float(sum(1.0 / (60 + v) for v in rr))
            row["n_channels_hit"] = float(len(rr))
            row["s0"] = s0map.get(c, 0.0)
            row["s0_rank"] = s0_rank[c]
            row["s0_gap_top"] = (scored[0][0] - row["s0"]) if scored else 0.0
            row["n_cands"] = n_codes
            row["ent_has_abbr"] = ent_has_abbr
            row["n_variants"] = len(vs)
            # string similarity on best description
            row["tset_best"] = best_sc
            row["best_is_first_desc"] = float(best_d == dn[0])
            row["tsort_orig"] = fuzz.token_sort_ratio(ent, best_d)
            row["ratio_orig"] = fuzz.ratio(ent, best_d)
            row["partial_orig"] = fuzz.partial_ratio(ent, best_d)
            row["wratio_best"] = fuzz.WRatio(best_v, best_d)
            row["best_from_variant"] = float(best_v != ent)
            dtoks = [t for t in best_d.split() if t not in STOPWORDS]
            all_dtoks = set()
            for d in dn:
                all_dtoks.update(d.split())
            all_dtoks -= set(STOPWORDS)
            # coverage / over- and under-coding signals
            cov_w = sum(w for t, w in zip(etoks, ew) if token_supported(t, dtoks))
            row["cov_rel_w"] = cov_w / ew_sum
            row["cov_rel_w_any"] = sum(w for t, w in zip(etoks, ew) if token_supported(t, list(all_dtoks))) / ew_sum
            row["drop_rel_w"] = sum(w for t, w in zip(etoks, ew) if not token_supported(t, dtoks))
            row["drop_max_w"] = max([w for t, w in zip(etoks, ew) if not token_supported(t, dtoks)], default=0.0)
            vt_list = list(vtok_union)
            extra = [t for t in dtoks if not token_supported(t, vt_list) and not t.isdigit()]
            row["cand_extra_tokens"] = len(extra)
            row["cand_extra_in_ctx"] = sum(t in wtoks for t in extra)
            row["cand_extra_in_chunk"] = sum(t in chunk_toks for t in extra)
            row["cand_extra_not_ctx"] = row["cand_extra_tokens"] - row["cand_extra_in_chunk"]
            row["cand_extra_unspec"] = sum(t in UNSPEC_WORDS for t in extra)
            row["desc_len"] = len(dtoks)
            row["ent_len"] = len(etoks)
            row["len_diff"] = len(dtoks) - len(etoks)
            row["n_desc"] = math.log1p(len(kb.descs_norm[c]))
            # context support of the candidate beyond the entity itself
            ctx_only = [t for t in all_dtoks if t in wtoks and t not in lit_toks]
            row["chunk_overlap"] = sum(1 for t in all_dtoks if t in chunk_toks and t not in lit_toks)
            row["ctx_overlap"] = len(ctx_only)
            row["ctx_overlap_frac"] = len(ctx_only) / max(1, len(all_dtoks))
            row["ctx_found"] = {"exact": 2.0, "fuzzy": 1.0}.get(ctx.found, 0.0)
            row["ctx_empty"] = float(ctx.found == "empty")
            # qualifiers
            qc = extract(best_d)
            for k, v in compare(qual_e, qc).items():
                row[k] = v
            row["q_lat_ctx_match"] = float(qual_w.lat is not None and qual_w.lat == qc.lat)
            row["q_lat_ctx_contra"] = float(qual_w.lat is not None and qc.lat is not None and qual_w.lat != qc.lat)
            # cues x candidate type
            cand_hist = float(any(w in best_d.split() for w in ("history",)))
            row["cue_neg"], row["cue_hist"] = float(ctx.negated), float(ctx.history)
            row["cue_unc"], row["cue_fam"] = float(ctx.uncertain), float(ctx.family)
            row["cand_history_word"] = cand_hist
            row["hist_x_candhist"] = float(ctx.history) * cand_hist
            row["unspec_desc"] = float(any(w in best_d.split() for w in UNSPEC_WORDS))
            row["other_desc"] = float("other" in best_d.split())
            row["code_freq"] = 0.0 if c == excl else math.log1p(tres.code_freq.get(c, 0))
            if system == "ICD10CM":
                nd = icd_nodot(c)
                row["prior_letter_aff"] = tres.label_affinity("letter", etoks, c[0])
                is_unsp = str(int(any(w in dn[0].split() for w in ("unspecified", "nos", "other"))))
                row["prior_unspec_aff"] = tres.label_affinity("unspec", etoks, is_unsp)
                row["prior_depth_aff"] = tres.label_affinity("depth", etoks, str(len(nd)))
                par = kb.parent.get(c)
                row["depth"] = len(nd)
                row["is_leaf"] = float(kb.is_leaf.get(c, True))
                row["is_cat3"] = float(len(nd) == 3)
                row["n_children"] = len(kb.children.get(c, []))
                row["n_siblings"] = len(kb.children.get(par, [])) if par else 0
                row["last_char_9"] = float(nd[-1] == "9")
                row["last_char_0"] = float(nd[-1] == "0")
                row["is_z"] = float(c[0] == "Z")
                row["is_r"] = float(c[0] == "R")
                row["depth_prior"] = tres.depth_prior.get(len(nd), 0.0)
                row["letter_prior"] = tres.letter_prior.get(c[0], 0.0)
                # training support of the 3-char category (excluding the dropped gold code itself)
                cf = tres.cat_freq.get(nd[:3], 0) - (tres.code_freq.get(c, 0) if c == excl else 0)
                row["cat_train_freq"] = math.log1p(max(0, cf))
                row["cat_mass"] = cat_mass[kb.category(c)] / tot_mass if tot_mass else 0.0
                row["cat_rank"] = cat_rank.get(kb.category(c), 99)
                row["is_anc_of_top"] = float(c in top_anc)
                row["is_desc_of_top"] = float(top_code is not None and top_code in kb.ancestors(c))
                row["is_top"] = float(c == top_code)
                row["is_7char_ext"] = float(len(nd) == 7)
                row["ext_A"] = float(len(nd) == 7 and nd[-1] == "A")
            else:
                tag = kb.sem_tag.get(c, "unknown")
                row["tag_prior"] = tres.tag_prior.get(tag, 0.0)
                row["prior_tag_aff"] = tres.label_affinity("tag", etoks, tag)
                for t in TOP_TAGS:
                    row[f"tag_{t.replace(' ', '_')}"] = float(tag == t)
            # character cosine between original entity and best description
            out.append((row, best_d))
        # vectorized char cosine
        if out:
            D = lex.char_vec([bd or "_" for _, bd in out])
            cos = (D @ e_char.T).toarray().ravel()
            if ctx.window:
                w_char = lex.char_vec([ctx.window])
                ccos = (D @ w_char.T).toarray().ravel()
            else:
                ccos = np.zeros(len(out))
            for (row, bd), cs, cc in zip(out, cos, ccos):
                row["char_cos_best"] = float(cs)
                row["ctx_char_cos"] = float(cc)  # context-window vs candidate description
        rows = []
        for row, _ in out:
            if gold is not None:
                row["label"] = int(row["code"] == gold)
            rows.append(row)
        return rows
