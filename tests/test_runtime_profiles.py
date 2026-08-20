from __future__ import annotations

import unittest

from work_agent_core.runtime_profiles import (
    FRIDAY_CONVERSATION_ID,
    RuntimeProfile,
    RuntimeProfileRegistry,
    TASK_PROFILE,
    friday_profile,
)


class RuntimeProfileResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = RuntimeProfileRegistry([TASK_PROFILE, friday_profile("Friday")])

    def test_the_persistent_runtime_claims_its_own_conversation(self) -> None:
        self.assertEqual(self.registry.resolve(FRIDAY_CONVERSATION_ID).id, "friday")

    def test_every_other_conversation_falls_back_to_the_task_runtime(self) -> None:
        for conversation_id in ("local-123", "", "friday-main-2"):
            self.assertEqual(self.registry.resolve(conversation_id).id, "task", conversation_id)

    def test_ordinary_chats_carry_memory_and_reminders(self) -> None:
        """The capabilities used to be implied by an id comparison.

        Memory extraction and reminder creation were switched off in every chat
        except one, as a side effect of that comparison rather than a decision.
        """

        task = self.registry.resolve("local-123")
        self.assertTrue(task.memory_enabled)
        self.assertTrue(task.reminders_enabled)
        # A persona's unprompted messages stay with the persona.
        self.assertFalse(task.proactive_messages)
        self.assertTrue(self.registry.resolve(FRIDAY_CONVERSATION_ID).proactive_messages)

    def test_a_new_runtime_needs_registration_not_a_new_call_site(self) -> None:
        night = RuntimeProfile(
            id="night-shift",
            label="夜间值守",
            conversation_ids=frozenset({"night-1"}),
            proactive_messages=True,
            priority=50,
        )
        self.registry.register(night)
        self.assertEqual(self.registry.resolve("night-1").id, "night-shift")
        self.assertEqual(self.registry.resolve(FRIDAY_CONVERSATION_ID).id, "friday")

    def test_highest_priority_claim_wins(self) -> None:
        greedy = RuntimeProfile(
            id="greedy",
            label="抢占",
            conversation_ids=frozenset({FRIDAY_CONVERSATION_ID}),
            priority=999,
        )
        self.registry.register(greedy)
        self.assertEqual(self.registry.resolve(FRIDAY_CONVERSATION_ID).id, "greedy")


if __name__ == "__main__":
    unittest.main()
