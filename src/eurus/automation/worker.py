from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import Awaitable, Callable
from typing import TypeAlias

from eurus.storage.jobs import Job, JobStore

JobHandler: TypeAlias = Callable[[Job], Awaitable[str]]
FinishHandler: TypeAlias = Callable[[Job, str, str, str | None], None]


class AutomationWorkerPool:
    def __init__(
        self,
        store: JobStore,
        handler: JobHandler,
        *,
        workers: int = 2,
        poll_seconds: float = 1.0,
        lease_seconds: int = 60,
        on_finish: FinishHandler | None = None,
    ) -> None:
        if not 1 <= workers <= 64:
            raise ValueError("workers must be between 1 and 64")
        if not 0.05 <= poll_seconds <= 300:
            raise ValueError("poll_seconds must be between 0.05 and 300")
        if not 5 <= lease_seconds <= 86400:
            raise ValueError("lease_seconds must be between 5 and 86400")
        self.store = store
        self.handler = handler
        self.workers = workers
        self.poll_seconds = poll_seconds
        self.lease_seconds = lease_seconds
        self.on_finish = on_finish
        self._stopping = asyncio.Event()

    async def run_once(self, worker_id: str | None = None) -> bool:
        identifier = worker_id or f"worker-{uuid.uuid4().hex}"
        job = self.store.claim(identifier, lease_seconds=self.lease_seconds)
        if job is None:
            return False
        if not self.store.mark_side_effect_started(job.job_id, identifier, job.lease_version):
            return True
        task: asyncio.Future[str] = asyncio.ensure_future(self.handler(job))
        heartbeat = asyncio.create_task(self._heartbeat(job, identifier, task))
        status = "completed"
        result = ""
        error: str | None = None
        try:
            result = await task
        except asyncio.CancelledError:
            status = "cancelled"
            error = "worker_cancelled"
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise
        except Exception as exc:
            status = "failed"
            error = type(exc).__name__
            result = f"Job failed: {error}"
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
        committed = self.store.finish(
            job.job_id,
            identifier,
            job.lease_version,
            status=status,
            result=result,
            error=error,
        )
        if committed and self.on_finish is not None:
            self.on_finish(job, status, result, error)
        return True

    async def run_forever(self, worker_id: str | None = None) -> None:
        identifier = worker_id or f"worker-{uuid.uuid4().hex}"
        while not self._stopping.is_set():
            self.store.recover_expired()
            worked = await self.run_once(identifier)
            if not worked:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stopping.wait(), timeout=self.poll_seconds)

    async def run(self) -> None:
        tasks = [
            asyncio.create_task(self.run_forever(f"worker-{index + 1}"))
            for index in range(self.workers)
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            self._stopping.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self) -> None:
        self._stopping.set()

    async def _heartbeat(self, job: Job, worker_id: str, task: asyncio.Future[str]) -> None:
        interval = max(1.0, self.lease_seconds / 3)
        while not task.done():
            await asyncio.sleep(interval)
            if task.done():
                return
            if not self.store.heartbeat(
                job.job_id, worker_id, job.lease_version, lease_seconds=self.lease_seconds
            ):
                task.cancel()
                return
