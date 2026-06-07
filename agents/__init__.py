"""Track D: real tool-using agents governed by OpenMirror.

The reasoning brain is pluggable (any OpenAI-compatible endpoint). Tools are
real and executable. Acting is governed at the weight level by the control
plane: ``/act`` returns ``allowed_calls`` / ``blocked_calls``, the loop only
executes the allowed ones.

Layout:
    brain.py         OpenAI-compatible client (vLLM local by default)
    tools.py         Real no-key tools (+ optional key tools) + registry
    loop.py          Governed ReAct loop (one agent, one principal)
    orchestrator.py  Planner → research-agent / ops-agent / support-agent
    workers.py       Role roster + default skill grants
"""
