"""Source metadata cannot turn an escaped HTML link into executable content."""

import pandas as pd
import pytest

from filingshock.ui.components import source_link


@pytest.mark.parametrize("url", [None, "", pd.NA, float("nan"), "javascript:alert(1)",
                                 "data:text/html,<script>alert(1)</script>", "file:///etc/passwd",
                                 "//example.com/path", "https:///missing-host", "https://[bad",
                                 "java\nscript:alert(1)", "https://example.com/\x00"])
def test_unsafe_or_malformed_source_urls_are_not_links(url):
    rendered = source_link(url, "<Source>")
    assert "<a " not in rendered
    assert rendered == "&lt;Source&gt; (source link unavailable)"


@pytest.mark.parametrize("scheme", ["http", "https", "HTTPS"])
def test_web_sources_remain_escaped_clickable_links(scheme):
    rendered = source_link(f'{scheme}://www.sec.gov/doc.htm?a=1&b="2"', "<Source>")
    assert "<a " in rendered
    assert '&amp;b=&quot;2&quot;' in rendered
    assert "&lt;Source&gt;" in rendered
    assert 'rel="noopener noreferrer"' in rendered
