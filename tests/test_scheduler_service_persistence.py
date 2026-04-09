from unittest.mock import AsyncMock, Mock

import pytest

from app.services.scheduler_service import SchedulerService


@pytest.mark.asyncio
async def test_pause_and_resume_job_persist_paused_state():
    scheduler = Mock()
    service = SchedulerService(scheduler)

    db = Mock()
    db.scheduler_metadata.update_one = AsyncMock()
    service.db = db

    assert await service.pause_job("job-1") is True
    scheduler.pause_job.assert_called_once_with("job-1")
    db.scheduler_metadata.update_one.assert_awaited_with(
        {"job_id": "job-1"},
        {"$set": {"job_id": "job-1", "paused": True, "updated_at": db.scheduler_metadata.update_one.await_args.args[1]["$set"]["updated_at"]}},
        upsert=True,
    )

    scheduler.reset_mock()
    db.scheduler_metadata.update_one.reset_mock()

    assert await service.resume_job("job-1") is True
    scheduler.resume_job.assert_called_once_with("job-1")
    db.scheduler_metadata.update_one.assert_awaited_with(
        {"job_id": "job-1"},
        {"$set": {"job_id": "job-1", "paused": False, "updated_at": db.scheduler_metadata.update_one.await_args.args[1]["$set"]["updated_at"]}},
        upsert=True,
    )


class _AsyncCursor:
    def __init__(self, docs):
        self._docs = iter(docs)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._docs)
        except StopIteration:
            raise StopAsyncIteration


@pytest.mark.asyncio
async def test_apply_persisted_job_states_restores_paused_jobs():
    running_job = Mock()
    running_job.next_run_time = object()

    paused_job = Mock()
    paused_job.next_run_time = None

    scheduler = Mock()
    scheduler.get_job.side_effect = lambda job_id: {
        "job-running": running_job,
        "job-paused": paused_job,
    }.get(job_id)

    service = SchedulerService(scheduler)

    db = Mock()
    db.scheduler_metadata.find.return_value = _AsyncCursor([
        {"job_id": "job-running", "paused": True},
        {"job_id": "job-paused", "paused": False},
    ])
    service.db = db

    await service.apply_persisted_job_states()

    scheduler.pause_job.assert_called_once_with("job-running")
    scheduler.resume_job.assert_called_once_with("job-paused")
