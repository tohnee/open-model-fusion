"""
open_fusion.tools - phase->toolset gating + tool execution.

Two responsibilities:
  1. `toolset_for_phase` decides which tools each phase is *allowed* to use. This is
     a correctness boundary, not a convenience: SYNTHESIS must have NO web tools so
     the evidence the judge reasoned over is frozen before the answer is written.
  2. `execute_tool` actually runs a tool call. Network access is lazy and every
     entry point honours `excluded_domains` *before* touching the network, so a
     blocked domain is deterministic and offline.

Tools available:
  - web_fetch   : stdlib urllib, no key needed.
  - web_search  : needs EXA_API_KEY (preferred) or BRAVE_API_KEY; otherwise returns
                  a 'not configured' result and the panel proceeds without it.
  - bash        : DISABLED unless OPEN_FUSION_ENABLE_BASH=1 (runs model-authored
                  shell locally; only enable inside a sandbox you trust).
"""
from __future__ import annotations

import asyncio
import json
import os
import shlex
import socket
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .schema import Phase


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


# ---- tool definitions (OpenAI JSON-schema style parameters) -----------------
WEB_SEARCH = ToolSpec(
    name="web_search",
    description="Search the web for current information. Returns a list of result "
                "snippets with titles and URLs.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "num_results": {"type": "integer", "description": "Max results (default 5)."},
        },
        "required": ["query"],
    },
)

WEB_FETCH = ToolSpec(
    name="web_fetch",
    description="Fetch the text content of a single URL.",
    parameters={
        "type": "object",
        "properties": {"url": {"type": "string", "description": "Absolute http(s) URL."}},
        "required": ["url"],
    },
)

BASH = ToolSpec(
    name="bash",
    description="Run a shell command locally and capture stdout/stderr. Disabled "
                "unless explicitly enabled by the operator.",
    parameters={
        "type": "object",
        "properties": {"command": {"type": "string", "description": "Shell command."}},
        "required": ["command"],
    },
)


def toolset_for_phase(phase: Phase) -> tuple[ToolSpec, ...]:
    """The phase->toolset gate. These exact sizes are asserted by the test suite
    and are a load-bearing design rule, not a default that may drift."""
    if phase is Phase.PANEL:
        return (WEB_SEARCH, WEB_FETCH, BASH)      # 3: panels may ground + compute
    if phase is Phase.JUDGE:
        return (WEB_SEARCH, WEB_FETCH)            # 2: judge may verify, never shell out
    if phase is Phase.SYNTHESIS:
        return ()                                 # 0: evidence is frozen here
    return ()


# ---- execution --------------------------------------------------------------
def _host_excluded(url: str, excluded: list[str]) -> bool:
    if not excluded:
        return False
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    for dom in excluded:
        d = dom.strip().lower()
        if d and (host == d or host.endswith("." + d)):
            return True
    return False


def _is_private_ip(host: str) -> bool:
    """Check if host resolves to a private/loopback/link-local address (SSRF guard)."""
    import ipaddress
    try:
        addr_infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for family, _, _, _, sockaddr in addr_infos:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if (ip.is_loopback or ip.is_private or ip.is_link_local or
                ip.is_multicast or ip.is_reserved or ip.is_unspecified):
            return True
    return False


def _validate_fetch_url(url: str) -> str | None:
    """Validate URL for safety; returns error message or None if ok."""
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return "invalid url"
    if parsed.scheme not in ("http", "https"):
        return f"blocked scheme: {parsed.scheme} (only http/https allowed)"
    host = parsed.hostname
    if not host:
        return "missing host"
    host_l = host.lower()
    if host_l in ("localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"):
        return f"blocked host: {host}"
    if _is_private_ip(host):
        return f"blocked private/loopback address: {host}"
    return None


async def execute_tool(name: str, arguments: dict[str, Any], *,
                       excluded_domains: list[str] | None = None) -> dict[str, Any]:
    """Dispatch one tool call. Always returns a JSON-serialisable dict with an `ok`
    flag; never raises (a tool failure is data the model should see, not a crash)."""
    excluded = list(excluded_domains or [])
    try:
        if name == "web_fetch":
            return await _web_fetch(str(arguments.get("url", "")), excluded)
        if name == "web_search":
            return await _web_search(str(arguments.get("query", "")),
                                     int(arguments.get("num_results", 5) or 5), excluded)
        if name == "bash":
            return await _bash(str(arguments.get("command", "")))
        return {"ok": False, "error": f"unknown tool: {name}"}
    except Exception as e:  # belt-and-braces: a tool bug must not kill the agent loop
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "tool": name}


async def _web_fetch(url: str, excluded: list[str]) -> dict[str, Any]:
    if not url:
        return {"ok": False, "error": "no url"}
    # SSRF guard: scheme + private/loopback/link-local IP check BEFORE any DNS/network
    if err := _validate_fetch_url(url):
        return {"ok": False, "error": err, "url": url}
    # excluded-domain check happens BEFORE any network call -> deterministic offline.
    if _host_excluded(url, excluded):
        return {"ok": False, "error": "domain excluded", "url": url}

    def _do() -> dict[str, Any]:
        req = urllib.request.Request(url, headers={"User-Agent": "open-fusion/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            ctype = resp.headers.get("Content-Type", "")
            body = resp.read(2_000_000).decode("utf-8", errors="replace")
        return {"ok": True, "url": url, "content_type": ctype, "text": body[:20000]}

    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _do)
    except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
        return {"ok": False, "error": f"fetch failed: {e}", "url": url}


async def _web_search(query: str, n: int, excluded: list[str]) -> dict[str, Any]:
    if not query:
        return {"ok": False, "error": "no query"}
    exa = os.getenv("EXA_API_KEY")
    brave = os.getenv("BRAVE_API_KEY")
    loop = asyncio.get_running_loop()
    if exa:
        return await loop.run_in_executor(None, _exa_search, query, n, excluded, exa)
    if brave:
        return await loop.run_in_executor(None, _brave_search, query, n, brave)
    return {"ok": False, "error": "search not configured (set EXA_API_KEY or BRAVE_API_KEY)"}


def _exa_search(query: str, n: int, excluded: list[str], key: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": query, "numResults": max(1, n)}
    if excluded:
        payload["excludeDomains"] = excluded
    req = urllib.request.Request(
        "https://api.exa.ai/search",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-api-key": key}, method="POST")
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    results = [{"title": r.get("title"), "url": r.get("url"), "text": (r.get("text") or "")[:1000]}
               for r in (data.get("results") or [])]
    return {"ok": True, "provider": "exa", "results": results}


def _brave_search(query: str, n: int, key: str) -> dict[str, Any]:
    qs = urllib.parse.urlencode({"q": query, "count": max(1, n)})
    req = urllib.request.Request(
        f"https://api.search.brave.com/res/v1/web/search?{qs}",
        headers={"Accept": "application/json", "X-Subscription-Token": key})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    web = (data.get("web") or {}).get("results") or []
    results = [{"title": r.get("title"), "url": r.get("url"),
                "text": (r.get("description") or "")[:1000]} for r in web]
    return {"ok": True, "provider": "brave", "results": results}


async def _bash(command: str) -> dict[str, Any]:
    if os.getenv("OPEN_FUSION_ENABLE_BASH") != "1":
        return {"ok": False, "error": "bash disabled (set OPEN_FUSION_ENABLE_BASH=1 to enable)"}
    if not command.strip():
        return {"ok": False, "error": "empty command"}

    def _do() -> dict[str, Any]:
        proc = subprocess.run(["bash", "-lc", command], capture_output=True,
                              text=True, timeout=30)
        return {"ok": proc.returncode == 0, "returncode": proc.returncode,
                "stdout": proc.stdout[:10000], "stderr": proc.stderr[:4000],
                "command": shlex.join(shlex.split(command))[:500]}

    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _do)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "bash timed out"}
