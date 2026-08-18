"""Re-export of :class:`nekron.selection.ColumnSelection`.

Column selection is shared by the feature stage and the alignment stage, so the
class itself lives in :mod:`nekron.selection`. This module lets the feature
package keep importing it as a sibling; new code should import it from
:mod:`nekron.selection` directly, as :mod:`nekron.align.merge` does.
"""

from __future__ import annotations

from nekron.selection import ColumnSelection

__all__ = ["ColumnSelection"]
