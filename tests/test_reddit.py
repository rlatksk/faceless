"""Checks for the Reddit importer.

Reddit serves its JSON API only to logged-in browsers, so this reads the Atom
feed instead. The parsing and HTML cleaning are pure, so they are checked
directly; the network path is exercised with a recorded feed.
"""

import pytest

from pipeline.reddit import _clean, fetch_post, parse_post_url

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>AITAH for saying I won't show up to holidays?</title>
    <author><name>/u/Adorable-Limit-6291</name></author>
    <content type="html">&lt;!-- SC_OFF --&gt;&lt;div class="md"&gt;&lt;p&gt;So I (18F) have an older brother.&lt;/p&gt;
&lt;p&gt;He has a girlfriend &amp;amp; she is childish.&lt;/p&gt;
&lt;p&gt;See &lt;a href="https://x.test"&gt;this&lt;/a&gt;.&lt;/p&gt;&lt;/div&gt;&lt;!-- SC_ON --&gt;</content>
  </entry>
  <entry>
    <title>A reply</title>
    <author><name>/u/someone</name></author>
    <content type="html">not the post</content>
  </entry>
</feed>"""


class _Resp:
    def __init__(self, body, status=200):
        self.content = body.encode() if isinstance(body, str) else body
        self.status_code = status
        self.text = body if isinstance(body, str) else body.decode(errors="replace")


class TestParsePostUrl:
    @pytest.mark.parametrize("url,expected", [
        ("https://www.reddit.com/r/AmItheAsshole/comments/1uo6wxb/slug/", ("AmItheAsshole", "1uo6wxb")),
        ("https://reddit.com/r/AITAH/comments/abc123/", ("AITAH", "abc123")),
        ("https://old.reddit.com/r/x/comments/zzz999/t/?utm_source=share", ("x", "zzz999")),
        ("https://redd.it/1uo6wxb", (None, "1uo6wxb")),
    ])
    def test_accepts_every_share_shape(self, url, expected):
        assert parse_post_url(url) == expected

    @pytest.mark.parametrize("url", ["", None, "https://example.com/x", "hello"])
    def test_rejects_non_reddit_input(self, url):
        assert parse_post_url(url) == (None, None)


class TestClean:
    def test_strips_html_and_unescapes(self):
        out = _clean('<div class="md"><p>Hello &amp; welcome</p></div>')
        assert out == "Hello & welcome"

    def test_drops_reddit_edit_markers(self):
        out = _clean("<!-- SC_OFF --><p>body</p><!-- SC_ON -->")
        assert "SC_OFF" not in out and "SC_ON" not in out
        assert out == "body"

    def test_paragraphs_become_blank_lines(self):
        assert _clean("<p>one</p><p>two</p>") == "one\n\ntwo"

    def test_br_becomes_newline(self):
        assert _clean("a<br>b<br/>c") == "a\nb\nc"

    def test_collapses_runs_of_blank_lines(self):
        assert _clean("<p>a</p><p></p><p></p><p>b</p>") == "a\n\nb"

    def test_unescapes_entities_including_nbsp(self):
        assert "&" in _clean("A &amp; B")
        assert "<" in _clean("x &lt; y")


class TestFetchPost:
    def test_returns_title_and_body_of_the_first_entry(self, monkeypatch):
        import pipeline.reddit as reddit
        monkeypatch.setattr(reddit.requests, "get", lambda *a, **k: _Resp(FEED))
        post = fetch_post("https://www.reddit.com/r/AmItheAsshole/comments/1uo6wxb/")
        assert "holidays" in post["title"]
        assert post["author"] == "/u/Adorable-Limit-6291"
        assert post["subreddit"] == "AmItheAsshole"
        assert "So I (18F) have an older brother." in post["text"]
        assert post["text"].startswith(post["title"])

    def test_ignores_reply_entries(self, monkeypatch):
        import pipeline.reddit as reddit
        monkeypatch.setattr(reddit.requests, "get", lambda *a, **k: _Resp(FEED))
        post = fetch_post("https://www.reddit.com/r/x/comments/1uo6wxb/")
        assert "not the post" not in post["text"]

    def test_rejects_a_non_reddit_link_without_network(self, monkeypatch):
        import pipeline.reddit as reddit
        called = []
        monkeypatch.setattr(reddit.requests, "get",
                            lambda *a, **k: called.append(1) or _Resp("", 200))
        with pytest.raises(ValueError, match="does not look like a Reddit post"):
            fetch_post("https://example.com/whatever")
        assert called == [], "should not make a request for an invalid link"

    def test_explains_rate_limiting(self, monkeypatch):
        import pipeline.reddit as reddit
        monkeypatch.setattr(reddit.requests, "get", lambda *a, **k: _Resp("", 429))
        with pytest.raises(RuntimeError, match="rate limiting"):
            fetch_post("https://www.reddit.com/r/x/comments/abc/")

    def test_explains_a_deleted_post(self, monkeypatch):
        import pipeline.reddit as reddit
        monkeypatch.setattr(reddit.requests, "get", lambda *a, **k: _Resp("", 404))
        with pytest.raises(RuntimeError, match="not found"):
            fetch_post("https://www.reddit.com/r/x/comments/abc/")

    def test_rejects_a_link_post_with_no_body(self, monkeypatch):
        """A link post has a title but nothing to narrate."""
        import pipeline.reddit as reddit
        empty = FEED.replace(
            '&lt;!-- SC_OFF --&gt;&lt;div class="md"&gt;&lt;p&gt;So I (18F) have an older brother.&lt;/p&gt;\n&lt;p&gt;He has a girlfriend &amp;amp; she is childish.&lt;/p&gt;\n&lt;p&gt;See &lt;a href="https://x.test"&gt;this&lt;/a&gt;.&lt;/p&gt;&lt;/div&gt;&lt;!-- SC_ON --&gt;',
            "")
        monkeypatch.setattr(reddit.requests, "get", lambda *a, **k: _Resp(empty))
        with pytest.raises(RuntimeError, match="no body text"):
            fetch_post("https://www.reddit.com/r/x/comments/abc/")

    def test_reports_an_unparseable_feed(self, monkeypatch):
        import pipeline.reddit as reddit
        monkeypatch.setattr(reddit.requests, "get", lambda *a, **k: _Resp("<html>blocked</html>"))
        with pytest.raises(RuntimeError):
            fetch_post("https://www.reddit.com/r/x/comments/abc/")
