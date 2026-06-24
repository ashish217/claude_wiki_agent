"""Wikipedia-grounded question answering with Claude.

Public surface:
    from wiki_agent import answer_question, search_wikipedia
"""

from .agent import answer_question
from .wikipedia import search_wikipedia

__all__ = ["answer_question", "search_wikipedia"]
