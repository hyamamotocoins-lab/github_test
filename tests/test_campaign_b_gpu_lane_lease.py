"""CPU tests for exclusive GPU lane lease (campaign_b/_locks/gpu_lane.json)."""

from __future__ import annotations

import os
import socket
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.campaign_b.errors import GpuLaneHeldError
from src.campaign_b.execution_keys import (
    FOREIGN_HOST_STALE_HEARTBEAT_SEC,
    _hold_depth,
    acquire_gpu_lock,
    foreign_stale_heartbeat_sec,
    gpu_lane_lease,
    gpu_lane_path,
    read_gpu_lane_lease,
    refresh_gpu_lane_heartbeat,
    release_gpu_lock,
    try_reclaim_gpu_lane_lease,
)
from src.common import atomic_write_json


def _clear_depth(tmp_path: Path) -> None:
    _hold_depth.pop(str(tmp_path.resolve()), None)


def test_acquire_release_roundtrip(tmp_path: Path) -> None:
    _clear_depth(tmp_path)
    path = gpu_lane_path(tmp_path)
    assert not path.is_file()

    lease = acquire_gpu_lock(tmp_path, owner='test_owner')
    assert path.is_file()
    assert lease['owner'] == 'test_owner'
    assert lease['pid'] == os.getpid()
    assert lease['enforced'] is True
    assert lease['heartbeat_at']

    assert release_gpu_lock(tmp_path, owner='test_owner') is True
    assert not path.is_file()
    assert read_gpu_lane_lease(tmp_path) is None


def test_nested_acquire_same_process(tmp_path: Path) -> None:
    _clear_depth(tmp_path)
    with gpu_lane_lease(tmp_path, owner='outer'):
        assert gpu_lane_path(tmp_path).is_file()
        with gpu_lane_lease(tmp_path, owner='inner'):
            doc = read_gpu_lane_lease(tmp_path)
            assert doc is not None
            assert int(doc['depth']) == 2
        doc = read_gpu_lane_lease(tmp_path)
        assert doc is not None
        assert int(doc['depth']) == 1
    assert not gpu_lane_path(tmp_path).is_file()


def test_fail_closed_when_live_holder(tmp_path: Path) -> None:
    _clear_depth(tmp_path)
    path = gpu_lane_path(tmp_path)
    path.parent.mkdir(parents=True)
    now = datetime.now(timezone.utc).isoformat()
    # Foreign host + fresh heartbeat → cannot reclaim (fail closed).
    atomic_write_json(path, {
        'schema_version': 1,
        'key': 'gpu_lane',
        'owner': 'other_notebook',
        'pid': 1,
        'hostname': 'other-host-not-local',
        'acquired_at': now,
        'heartbeat_at': now,
        'depth': 1,
        'enforced': True,
    })
    with pytest.raises(GpuLaneHeldError, match='GPU lane lease held'):
        acquire_gpu_lock(tmp_path, owner='me', stale_heartbeat_sec=3600)
    assert path.is_file()


def test_reclaim_dead_pid(tmp_path: Path) -> None:
    _clear_depth(tmp_path)
    path = gpu_lane_path(tmp_path)
    path.parent.mkdir(parents=True)
    now = datetime.now(timezone.utc).isoformat()
    atomic_write_json(path, {
        'schema_version': 1,
        'key': 'gpu_lane',
        'owner': 'crashed',
        'pid': 2_147_483_647,  # almost certainly not alive
        'hostname': socket.gethostname(),
        'acquired_at': now,
        'heartbeat_at': now,
        'depth': 1,
        'enforced': True,
    })
    lease = acquire_gpu_lock(tmp_path, owner='reclaimer')
    assert lease['owner'] == 'reclaimer'
    assert lease['pid'] == os.getpid()
    assert 'reclaimed_from' in lease
    assert 'dead_pid' in str(lease['reclaimed_from'])
    release_gpu_lock(tmp_path, owner='reclaimer')
    assert not path.is_file()


def test_reclaim_stale_heartbeat(tmp_path: Path) -> None:
    _clear_depth(tmp_path)
    path = gpu_lane_path(tmp_path)
    path.parent.mkdir(parents=True)
    old = (datetime.now(timezone.utc) - timedelta(hours=10)).isoformat()
    atomic_write_json(path, {
        'schema_version': 1,
        'key': 'gpu_lane',
        'owner': 'stale_owner',
        'pid': 1,
        'hostname': 'foreign-machine',
        'acquired_at': old,
        'heartbeat_at': old,
        'depth': 1,
        'enforced': True,
    })
    lease = acquire_gpu_lock(tmp_path, owner='fresh', stale_heartbeat_sec=3600)
    assert lease['owner'] == 'fresh'
    assert 'stale_heartbeat' in str(lease.get('reclaimed_from'))
    release_gpu_lock(tmp_path, owner='fresh')


def test_foreign_host_stale_after_15m_reclaimable(tmp_path: Path) -> None:
    """Foreign host + heartbeat older than 15m → reclaimable (Paperspace switch)."""
    _clear_depth(tmp_path)
    path = gpu_lane_path(tmp_path)
    path.parent.mkdir(parents=True)
    old = (
        datetime.now(timezone.utc)
        - timedelta(seconds=FOREIGN_HOST_STALE_HEARTBEAT_SEC + 60)
    ).isoformat()
    atomic_write_json(path, {
        'schema_version': 1,
        'key': 'gpu_lane',
        'owner': 'old_machine',
        'pid': 13712,
        'hostname': 'n6plw4cgyr',
        'acquired_at': old,
        'heartbeat_at': old,
        'depth': 1,
        'enforced': True,
    })
    lease = acquire_gpu_lock(tmp_path, owner='new_machine')
    assert lease['owner'] == 'new_machine'
    assert 'stale_heartbeat_foreign_host' in str(lease.get('reclaimed_from'))
    release_gpu_lock(tmp_path, owner='new_machine')


def test_foreign_host_fresh_heartbeat_not_stale(tmp_path: Path) -> None:
    """Foreign host + fresh heartbeat → not reclaimable (fail closed)."""
    _clear_depth(tmp_path)
    path = gpu_lane_path(tmp_path)
    path.parent.mkdir(parents=True)
    # Age well under 15m but would be stale under a tiny same-host threshold.
    fresh = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    atomic_write_json(path, {
        'schema_version': 1,
        'key': 'gpu_lane',
        'owner': 'other_host',
        'pid': 99,
        'hostname': 'nvgqew8q38-other',
        'acquired_at': fresh,
        'heartbeat_at': fresh,
        'depth': 1,
        'enforced': True,
    })
    with pytest.raises(GpuLaneHeldError):
        acquire_gpu_lock(
            tmp_path,
            owner='me',
            stale_heartbeat_sec=1,  # must not affect foreign-host path
        )
    assert path.is_file()
    refused = try_reclaim_gpu_lane_lease(tmp_path)
    assert refused['action'] == 'refused'


def test_same_host_stale_heartbeat_after_15m_reclaimable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same host + heartbeat older than 15m → reclaimable (zombie kernel)."""
    _clear_depth(tmp_path)
    monkeypatch.setattr(
        'src.campaign_b.execution_keys._pid_alive',
        lambda _pid: True,
    )
    path = gpu_lane_path(tmp_path)
    path.parent.mkdir(parents=True)
    old = (
        datetime.now(timezone.utc)
        - timedelta(seconds=FOREIGN_HOST_STALE_HEARTBEAT_SEC + 60)
    ).isoformat()
    atomic_write_json(path, {
        'schema_version': 1,
        'key': 'gpu_lane',
        'owner': 'zombie_kernel',
        'pid': 16_650,
        'hostname': socket.gethostname(),
        'acquired_at': old,
        'heartbeat_at': old,
        'depth': 1,
        'enforced': True,
    })
    lease = acquire_gpu_lock(tmp_path, owner='fresh_start')
    assert lease['owner'] == 'fresh_start'
    assert 'stale_heartbeat' in str(lease.get('reclaimed_from'))
    release_gpu_lock(tmp_path, owner='fresh_start')


def test_same_host_dead_pid_reclaimable(tmp_path: Path) -> None:
    _clear_depth(tmp_path)
    path = gpu_lane_path(tmp_path)
    path.parent.mkdir(parents=True)
    now = datetime.now(timezone.utc).isoformat()
    atomic_write_json(path, {
        'schema_version': 1,
        'key': 'gpu_lane',
        'owner': 'dead',
        'pid': 2_147_483_647,
        'hostname': socket.gethostname(),
        'acquired_at': now,
        'heartbeat_at': now,
        'depth': 1,
        'enforced': True,
    })
    result = try_reclaim_gpu_lane_lease(tmp_path)
    assert result['action'] == 'reclaimed'
    assert 'dead_pid' in str(result.get('reason'))
    assert not path.is_file()


def test_refresh_updates_heartbeat_only(tmp_path: Path) -> None:
    _clear_depth(tmp_path)
    lease = acquire_gpu_lock(tmp_path, owner='holder')
    acquired = lease['acquired_at']
    old_hb = lease['heartbeat_at']
    depth = lease['depth']
    time.sleep(0.02)
    updated = refresh_gpu_lane_heartbeat(tmp_path)
    assert updated is not None
    assert updated['heartbeat_at'] != old_hb
    assert updated['acquired_at'] == acquired
    assert updated['owner'] == 'holder'
    assert updated['pid'] == os.getpid()
    assert updated['hostname'] == socket.gethostname()
    assert int(updated['depth']) == int(depth)
    release_gpu_lock(tmp_path, owner='holder')


def test_refresh_noop_when_not_holder(tmp_path: Path) -> None:
    _clear_depth(tmp_path)
    path = gpu_lane_path(tmp_path)
    path.parent.mkdir(parents=True)
    now = datetime.now(timezone.utc).isoformat()
    atomic_write_json(path, {
        'schema_version': 1,
        'key': 'gpu_lane',
        'owner': 'other',
        'pid': 1,
        'hostname': 'foreign-host',
        'acquired_at': now,
        'heartbeat_at': now,
        'depth': 1,
        'enforced': True,
    })
    assert refresh_gpu_lane_heartbeat(tmp_path) is None


def test_foreign_stale_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('VALIDATED_RG_GPU_LANE_FOREIGN_STALE_SEC', '120')
    assert foreign_stale_heartbeat_sec() == 120
    assert foreign_stale_heartbeat_sec(30) == 30
    monkeypatch.delenv('VALIDATED_RG_GPU_LANE_FOREIGN_STALE_SEC', raising=False)
    assert foreign_stale_heartbeat_sec() == FOREIGN_HOST_STALE_HEARTBEAT_SEC
