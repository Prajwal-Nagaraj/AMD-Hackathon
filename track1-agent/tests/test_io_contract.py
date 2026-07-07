import asyncio
import json

from agent import main as agent_main
from agent.fireworks import CallResult


class FakeClient:
    """Stands in for FireworksClient so tests never touch the network."""

    def __init__(self, text="ANSWER: 42", raise_for=None):
        self.text = text
        self.raise_for = raise_for or set()
        self.calls = 0

    async def complete(self, *, model, user, system=None, max_tokens, stop=None,
                        temperature=0.0, extra_body=None, timeout=25.0):
        self.calls += 1
        if model in self.raise_for:
            raise RuntimeError(f"simulated failure for {model}")
        return CallResult(
            text=self.text,
            prompt_tokens=5,
            completion_tokens=2,
            total_tokens=7,
            model=model,
        )


def test_run_all_produces_valid_results_for_every_task():
    tasks = [
        {"task_id": "t1", "prompt": "What is 2 + 2?"},
        {"task_id": "t2", "prompt": "What is the capital of France?"},
        {"task_id": "t3", "prompt": "Summarize the following text in one sentence: ..."},
    ]
    client = FakeClient()
    results = asyncio.run(agent_main.run_all(tasks, client=client))

    assert len(results) == 3
    assert [r["task_id"] for r in results] == ["t1", "t2", "t3"]
    for r in results:
        assert "answer" in r
        assert isinstance(r["answer"], str)


def test_run_all_never_crashes_on_a_failing_task():
    tasks = [{"task_id": "t1", "prompt": "What is 2 + 2?"}]
    client = FakeClient(raise_for={"gemma-4-31b-it"})
    results = asyncio.run(agent_main.run_all(tasks, client=client))

    assert results == [{"task_id": "t1", "answer": ""}]


def test_write_results_roundtrip(tmp_path):
    path = tmp_path / "results.json"
    data = [{"task_id": "t1", "answer": "ok"}]
    agent_main.write_results(path, data)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == data


def test_load_tasks_reads_json_array(tmp_path):
    path = tmp_path / "tasks.json"
    payload = [{"task_id": "t1", "prompt": "hi"}]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert agent_main.load_tasks(path) == payload
