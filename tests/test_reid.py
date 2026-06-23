"""Smoke test for reid.EmbeddingStore -- the appearance-clustering store (previously untested).

Inserts synthetic L2-normalized vectors straight into detection_embeddings (no GPU / model), then
checks that load + cluster() groups look-alikes. Guards the consolidation of the decode/cluster
primitives into reidutil (EmbeddingStore.cluster now delegates there).
"""
from __future__ import annotations

import numpy as np

import db
import reid


def _unit(*v):
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def _add_crop_with_vector(conn, vec, *, species="raccoon", conf=0.9):
    db.insert_detection(conn, timestamp=db.now_local_iso(), source="glass_door_cam",
                        detection_class="animal", confidence=conf, bbox=(0, 0, 10, 10),
                        frame_w=100, frame_h=100, crop_path="crops/x.jpg", species=species)
    did = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    u = _unit(*vec)
    db.insert_embedding(conn, did, "m", len(u), u.tobytes())


def test_embeddingstore_clusters_lookalikes_into_two_groups(conn):
    for _ in range(3):
        _add_crop_with_vector(conn, (1, 0, 0))     # cluster A
    for _ in range(2):
        _add_crop_with_vector(conn, (0, 1, 0))     # cluster B (orthogonal -> far in cosine)

    store = reid.EmbeddingStore(conn, "raccoon", 0.5, "m")
    assert len(store) == 5
    labels = store.cluster(threshold=0.3, method="average")
    assert len(labels) == 5
    assert len(set(labels)) == 2                    # the two orthogonal groups separate
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]


def test_embeddingstore_cluster_handles_single_crop(conn):
    _add_crop_with_vector(conn, (1, 0, 0))
    store = reid.EmbeddingStore(conn, "raccoon", 0.5, "m")
    assert list(store.cluster(threshold=0.3, method="average")) == [1]   # one trivial cluster
