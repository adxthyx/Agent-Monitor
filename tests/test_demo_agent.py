from datetime import UTC, datetime, timedelta

from demo_agent import app as demo


def set_faults(**kw):
    return demo.set_faults(demo.FaultPatch(**kw))


async def test_gradual_decay_corrupts_a_growing_share_of_facts():
    await demo.reset_faults()
    try:
        order = list(demo.KB["decayed"])
        f = await set_faults(gradual_decay=True, decay_per_hour=0.5)
        assert f.decay_started_at is not None and f.decayed_facts == []

        demo.faults = f.model_copy(update={"decay_started_at": datetime.now(UTC) - timedelta(hours=1)})
        half = demo.faults.decayed_facts
        assert half == order[: len(order) // 2]
        kb = demo.knowledge_base()
        assert demo.KB["decayed"][half[0]] in kb and demo.KB["facts"][half[0]] not in kb
        assert demo.KB["facts"][order[-1]] in kb  # canary fact decays last

        demo.faults = demo.faults.model_copy(update={"decay_started_at": datetime.now(UTC) - timedelta(hours=5)})
        assert demo.faults.decayed_facts == order  # capped at every fact

        restarted = await set_faults(gradual_decay=True)  # already on: the clock keeps running
        assert restarted.decay_started_at == demo.faults.decay_started_at
        off = await set_faults(gradual_decay=False)
        assert off.decay_started_at is None and off.decayed_facts == []
    finally:
        await demo.reset_faults()


def test_every_fact_has_a_decayed_variant():
    assert set(demo.KB["decayed"]) == set(demo.KB["facts"])
