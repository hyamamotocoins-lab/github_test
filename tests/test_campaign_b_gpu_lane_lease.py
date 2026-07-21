"""CPU tests for exclusive GPU lane lease (campaign_b/_locks/gpu_lane.json)."""

from __future__ import annotations

import os
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.campaign_b.errors import GpuLaneHeldError
from src.campaign_b.execution_keys import (
    _hold_depth,
    acquire_gpu_lock,
    gpu_lane_lease,
    gpu_lane_path,
    read_gpu_lane_lease,
    release_gpu_lock,
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
