"""OpenMirror memory track — log chats, consolidate into user_style adapters.

Production path: interactions → curate → POST /personalize (Track B → Track A).
Does NOT use WeaveSelf's separate trainer; see ``ml/weaveself/`` for research code.
"""
