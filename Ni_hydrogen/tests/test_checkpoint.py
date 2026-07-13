"""
tests/test_checkpoint.py
=========================
Unit tests for models/checkpoint.py -- the uniform .done sentinel marker
mechanism used across neb_workflow, diffusivity_workflow,
permeation_workflow, vibrations, and neb_subsurface.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.checkpoint import is_done, mark_done


class TestIsDone:
    def test_false_when_marker_absent(self, tmp_path):
        marker = tmp_path / 'step.done'
        assert is_done(marker) is False

    def test_true_when_marker_present(self, tmp_path):
        marker = tmp_path / 'step.done'
        marker.touch()
        assert is_done(marker) is True

    def test_accepts_str_path(self, tmp_path):
        marker = str(tmp_path / 'step.done')
        assert is_done(marker) is False
        open(marker, 'w').close()
        assert is_done(marker) is True


class TestMarkDone:
    def test_creates_empty_marker_file(self, tmp_path):
        marker = tmp_path / 'step.done'
        mark_done(marker)
        assert marker.exists()
        assert marker.read_text() == ''

    def test_creates_parent_directories(self, tmp_path):
        marker = tmp_path / 'a' / 'b' / 'c' / 'step.done'
        assert not marker.parent.exists()
        mark_done(marker)
        assert marker.exists()

    def test_idempotent_on_repeat_calls(self, tmp_path):
        marker = tmp_path / 'step.done'
        mark_done(marker)
        mark_done(marker)  # must not raise
        assert marker.exists()

    def test_accepts_str_path(self, tmp_path):
        marker = str(tmp_path / 'step.done')
        mark_done(marker)
        assert os.path.exists(marker)


class TestRoundTrip:
    def test_mark_then_is_done(self, tmp_path):
        marker = tmp_path / 'nested' / 'step.done'
        assert is_done(marker) is False
        mark_done(marker)
        assert is_done(marker) is True
