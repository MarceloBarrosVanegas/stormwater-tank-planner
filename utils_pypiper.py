# dir_tree.py
# ------------------------------------------------------------------
#   Clase sin CLI que:
#      1) Extrae **recursivamente** la estructura de una carpeta.
#      2) La reconstruye (solo directorios) en otra ubicación.
# ------------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
import os
import re
from typing import Iterable, List

_INDENT = 4  # espacios que representan un nivel (igual que tree /a)


class DirTree:
    # ==============================================================
    # 1)  EXPORTAR ÁRBOL  (depth-first, 100 % recursivo)
    # ==============================================================
    def export_tree(self, src: str | Path) -> str:
        """
        Devuelve un string con la jerarquía **completa** de *src*
        (idéntico orden pre-orden que `tree /a /ad`, pero sólo carpetas).
        """
        src = Path(src).resolve()
        if not src.is_dir():
            raise NotADirectoryError(src)

        lines: List[str] = [src.name]

        def _walk(folder: Path, level: int) -> None:
            subdirs = sorted([d for d in folder.iterdir() if d.is_dir()])
            for i, d in enumerate(subdirs):
                connector = "+---" if i < len(subdirs) - 1 else r"\---"
                lines.append(f"{' ' * (_INDENT * level)}{connector} {d.name}")
                _walk(d, level + 1)                       # <-- recursión

        _walk(src, 0)
        return "\n".join(lines)

    # ==============================================================
    # 2)  REPLICAR ÁRBOL  (interpreta los niveles por indentación)
    # ==============================================================
    def replicate_tree(
        self,
        tree_text: str | Iterable[str],
        dest: str | Path,
    ) -> None:
        """
        Crea la misma estructura (carpetas y subcarpetas) en *dest*.
        *tree_text* acepta:
          • el string devuelto por `export_tree`
          • un iterable de líneas (por ejemplo, f.readlines()).
        """
        if isinstance(tree_text, str):
            lines = tree_text.splitlines()
        else:
            lines = list(tree_text)

        dest = Path(dest).resolve()
        dest.mkdir(parents=True, exist_ok=True)

        stack: List[Path] = [dest]          # stack[ nivel ] = carpeta
        # Cabecera (línea 0) se ignora — es sólo el nombre raíz original.
        for raw in lines[1:]:
            if not raw.strip():
                continue

            # Nivel = espacios de indentación / INDENT
            indent_spaces = len(raw) - len(raw.lstrip(" "))
            level = indent_spaces // _INDENT

            # Quitar prefijos '+--- ' o '\--- '
            m = re.match(r"[+\\]---\s+(.*)", raw.strip())
            name = m.group(1) if m else raw.strip()

            # Ajustar pila al nivel correcto
            while len(stack) - 1 > level:
                stack.pop()

            parent = stack[-1]
            new_dir = parent / name
            new_dir.mkdir(exist_ok=True)

            # Añadir para ser padre de los siguientes hijos
            if len(stack) - 1 == level:
                stack.append(new_dir)
            else:  # (cuando la línea salta varios niveles de golpe)
                stack[-1] = new_dir

    # ==============================================================
    # 3)  ATAJOS convenientes (opcional)
    # ==============================================================
    def export_to_file(self, src: str | Path, out_file: str | Path) -> Path:
        """Guarda el árbol en un .txt y devuelve la ruta creada."""
        out_file = Path(out_file)
        out_file.write_text(self.export_tree(src), encoding="utf-8")
        return out_file

    def replicate_from_file(self, tree_file: str | Path, dest: str | Path) -> None:
        """Lee el árbol desde *tree_file* y lo recrea en *dest*."""
        self.replicate_tree(Path(tree_file).read_text(encoding="utf-8"), dest)
