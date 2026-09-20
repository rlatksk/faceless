"""Fetch a Reddit post's title and body for use as source text.

Reddit blocks its JSON API for non-browser clients — www.reddit.com/.json and
api.reddit.com both return 403, and old.reddit.com redirects to a login wall.
The Atom feed at /comments/<id>/.rss is served to plain HTTP clients, and its
first entry is the post itself, so that is what this uses. No API key, no OAuth,
no browser.
"""

import html
import re
import xml.etree.ElementTree as ET

import requests

# Reddit rate-limits by client and returns 429 without a browser-shaped UA.
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/atom+xml, application/xml, text/xml, */*",
}

_ATOM = {"a": "http://www.w3.org/2005/Atom"}

# Matches the two shapes a share link takes: the full comments URL and the
# redd.it short form. The subreddit is optional because redd.it omits it.
_URL_RE = re.compile(
    r"(?:reddit\.com/r/(?P<sub>[^/?#]+)/comments/(?P<id>[a-z0-9]+)"
    r"|redd\.it/(?P<short>[a-z0-9]+))",
    re.IGNORECASE,
)


def parse_post_url(url):
    """Return (subreddit, post_id) from a Reddit link, or (None, None).

    The subreddit may be None for a redd.it short link, which still resolves —
    the RSS endpoint only needs the post id.
    """
    if not url:
        return None, None
    match = _URL_RE.search(url)
    if not match:
        return None, None
    return match.group("sub"), match.group("id") or match.group("short")


def _clean(body):
    """Turn a Reddit HTML body into plain text."""
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)   # SC_OFF / SC_ON markers
    body = re.sub(r"<br\s*/?>", "\n", body, flags=re.I)
    body = re.sub(r"</p>", "\n\n", body, flags=re.I)
    body = re.sub(r"<[^>]+>", "", body)
    body = html.unescape(body)
    body = re.sub(r"[ \t]+\n", "\n", body)
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def fetch_post(url):
    """Fetch a Reddit post as {"title", "author", "subreddit", "text"}.

    Raises ValueError for a link that is not a Reddit post, and RuntimeError for
    anything the network or Reddit refuses.
    """
    sub, post_id = parse_post_url(url)
    if not post_id:
        raise ValueError(
            "That does not look like a Reddit post link. Paste the post URL, "
            "for example https://www.reddit.com/r/AmItheAsshole/comments/abc123/title/"
        )

    path = f"/r/{sub}" if sub else ""
    endpoint = f"https://www.reddit.com{path}/comments/{post_id}/.rss"
    try:
        resp = requests.get(endpoint, headers=_HEADERS, timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(f"Could not reach Reddit: {e}")

    if resp.status_code == 429:
        raise RuntimeError("Reddit is rate limiting requests. Wait a minute and try again.")
    if resp.status_code == 404:
        raise RuntimeError("That Reddit post was not found. It may have been deleted.")
    if resp.status_code != 200:
        raise RuntimeError(f"Reddit returned HTTP {resp.status_code}.")

    try:
        entries = ET.fromstring(resp.content).findall("a:entry", _ATOM)
    except ET.ParseError as e:
        raise RuntimeError(f"Reddit returned something that is not a feed: {e}")

    if not entries:
        raise RuntimeError("That Reddit post has no content — it may have been removed.")

    entry = entries[0]     # the post itself; replies follow
    title = (entry.findtext("a:title", default="", namespaces=_ATOM) or "").strip()
    author = (entry.findtext("a:author/a:name", default="", namespaces=_ATOM) or "").strip()
    body = _clean(entry.findtext("a:content", default="", namespaces=_ATOM) or "")

    if not body:
        # Link posts have a title but no self text; the title alone is too thin
        # to write a video from, so say so rather than producing a bad script.
        raise RuntimeError(
            "That post has no body text — it is a link post. Reddit story posts "
            "with the story written out are what this needs."
        )

    text = f"{title}\n\n{body}" if title else body
    return {"title": title, "author": author, "subreddit": sub or "", "text": text}
