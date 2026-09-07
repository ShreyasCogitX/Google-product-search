"""Test queueing behavior of incoming requests."""
import asyncio
import time
import pytest
from httpx import AsyncClient
import main

@pytest.mark.asyncio
async def test_request_queue_fifo_processing(monkeypatch):
    execution_timeline = []

    def mock_product_search(query=None, queries=None, limit=5):
        q = query or (queries[0] if queries else "unknown")
        start_ts = time.time()
        execution_timeline.append(("START", q, start_ts))
        time.sleep(0.08)  # simulate processing
        end_ts = time.time()
        execution_timeline.append(("END", q, end_ts))
        return {
            "query": q,
            "count": 1,
            "products": [],
            "error": None,
            "source": "mock"
        }

    monkeypatch.setattr(main, "google_product_search", mock_product_search)

    async with AsyncClient(app=main.app, base_url="http://test") as client:
        # Fire 3 requests concurrently
        t0 = time.time()
        tasks = [
            client.post("/api/search", json={"text": f"request_{i}", "limit": 2})
            for i in range(1, 4)
        ]
        responses = await asyncio.gather(*tasks)

    # 1. All responses succeeded
    for resp in responses:
        assert resp.status_code == 200

    # 2. Verify all requests started and finished
    assert len(execution_timeline) == 6

    # 3. Verify strict sequential FIFO ordering:
    # req 1 START -> req 1 END -> req 2 START -> req 2 END -> req 3 START -> req 3 END
    action0, q0, t_start0 = execution_timeline[0]
    action1, q1, t_end0 = execution_timeline[1]
    action2, q2, t_start1 = execution_timeline[2]
    action3, q3, t_end1 = execution_timeline[3]
    action4, q4, t_start2 = execution_timeline[4]
    action5, q5, t_end2 = execution_timeline[5]

    assert (action0, q0) == ("START", "request_1")
    assert (action1, q1) == ("END", "request_1")
    assert (action2, q2) == ("START", "request_2")
    assert (action3, q3) == ("END", "request_2")
    assert (action4, q4) == ("START", "request_3")
    assert (action5, q5) == ("END", "request_3")

    # Ensure req 2 did not start until req 1 finished, and req 3 until req 2 finished
    assert t_start1 >= t_end0
    assert t_start2 >= t_end1
