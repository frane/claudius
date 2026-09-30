import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import claudius.tutor as tutor  # noqa: E402
from claudius.web import server  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "_load", lambda: server._state.update(model=object(), tok=object()))
    monkeypatch.setattr(tutor, "respond", lambda m, t, turns, level: {
        "text": "QUID TU HODIE FACIS?",
        "correction": {"kind": "firm", "wrong": "AMANT", "right": "AMAT",
                       "rule": "PUELLA NUMERI SINGULARIS EST, ERGO ET UERBUM: AMAT.",
                       "text": 'SED DICENDUM EST "AMAT", NON "AMANT".'},
        "full": 'SED DICENDUM EST "AMAT", NON "AMANT". QUID TU HODIE FACIS?'})
    with TestClient(server.app) as c:
        yield c


def test_reply_shows_roman_form_and_keeps_internal_history(client):
    r = client.post("/api/reply", json={"level": "tiro",
                                        "history": [{"role": "DISCIPULUS", "text": "PUELLA ROSAM AMANT."}]})
    assert r.status_code == 200
    j = r.json()
    assert j["text"] == "QVID TV HODIE FACIS?"
    assert j["correction"]["kind"] == "firm" and "VERBVM" in j["correction"]["rule"]
    assert "UERBUM" not in j["full"] and "DICENDUM" in j["full"], "history stays in internal U form"


def test_rejects_bad_requests(client):
    assert client.post("/api/reply", json={"level": "x", "history": [{"role": "DISCIPULUS", "text": "A"}]}).status_code == 400
    assert client.post("/api/reply", json={"level": "tiro", "history": [{"role": "MAGISTER", "text": "A"}]}).status_code == 400
    long = "A" * 301
    assert client.post("/api/reply", json={"level": "tiro", "history": [{"role": "DISCIPULUS", "text": long}]}).status_code == 400


def test_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "CLAVDIVS" in r.text and "MITTE" in r.text
