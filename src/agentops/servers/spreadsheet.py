"""Spreadsheet tools backed by a JSON workbook in AGENTOPS_DATA_DIR."""

from __future__ import annotations

import os
import re

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from agentops.servers.store import load_json, save_json

mcp = FastMCP(
    "spreadsheet",
    instructions="Read and update workbook sheets. Writes change the workbook file.",
    log_level="ERROR",
)

_READ = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
_WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=False)

SEED = {
    "budget": [
        ["quarter", "revenue", "cost"],
        ["Q1", "120", "80"],
        ["Q2", "140", "90"],
        ["Q3", "160", "95"],
    ],
    "tracker": [
        ["item", "owner", "status"],
    ],
}

_RANGE = re.compile(r"^([A-Z]{1,3})(\d+)(?::([A-Z]{1,3})(\d+))?$")
_SHEET = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,40}$")

# Remaining forced failures for this process. None means "read the env once".
_failures_left: int | None = None


def _workbook() -> dict:
    return load_json("workbook.json", SEED)


def _save(book: dict) -> None:
    save_json("workbook.json", book)


def _check_sheet(sheet: str) -> None:
    if not _SHEET.fullmatch(sheet):
        raise ValueError("invalid sheet name")


def _column(name: str) -> int:
    number = 0
    for char in name:
        number = number * 26 + (ord(char) - 64)
    return number


def _parse_range(cell_range: str) -> tuple[int, int, int, int]:
    match = _RANGE.fullmatch(cell_range.strip().upper())
    if not match:
        raise ValueError(f"unsupported range: {cell_range}")
    start_col, start_row, end_col, end_row = match.groups()
    col1 = _column(start_col)
    row1 = int(start_row)
    col2 = _column(end_col or start_col)
    row2 = int(end_row or start_row)
    if row1 < 1 or col1 < 1 or row2 < row1 or col2 < col1:
        raise ValueError(f"unsupported range: {cell_range}")
    return col1, row1, col2, row2


def _pad(grid: list[list[str]], rows: int, cols: int) -> None:
    while len(grid) < rows:
        grid.append([])
    for row in grid:
        while len(row) < cols:
            row.append("")


def _flake() -> None:
    """Fail the first N writes in this process when the env hook is set."""
    global _failures_left
    if _failures_left is None:
        _failures_left = int(os.environ.get("AGENTOPS_SPREADSHEET_FAIL_FIRST", "0"))
    if _failures_left > 0:
        _failures_left -= 1
        raise RuntimeError("transient spreadsheet backend error")


@mcp.tool(annotations=_READ)
def read_range(sheet: str, cell_range: str) -> dict:
    """Read a rectangular range. Missing cells come back as empty strings."""
    _check_sheet(sheet)
    col1, row1, col2, row2 = _parse_range(cell_range)
    grid = _workbook().get(sheet, [])
    values: list[list[str]] = []
    for row_index in range(row1, row2 + 1):
        line: list[str] = []
        for col_index in range(col1, col2 + 1):
            if row_index - 1 < len(grid) and col_index - 1 < len(grid[row_index - 1]):
                line.append(str(grid[row_index - 1][col_index - 1]))
            else:
                line.append("")
        values.append(line)
    return {"sheet": sheet, "cell_range": cell_range.strip().upper(), "values": values}


@mcp.tool(annotations=_WRITE)
def write_range(sheet: str, cell_range: str, values: list[list[str]]) -> dict:
    """Overwrite a range. The value grid must match the range shape."""
    _check_sheet(sheet)
    col1, row1, col2, row2 = _parse_range(cell_range)
    expected_rows = row2 - row1 + 1
    expected_cols = col2 - col1 + 1
    if len(values) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {len(values)}")
    for row in values:
        if len(row) != expected_cols:
            raise ValueError(f"expected {expected_cols} columns, got {len(row)}")
    _flake()
    book = _workbook()
    grid = book.setdefault(sheet, [])
    _pad(grid, row2, col2)
    for row_offset, row in enumerate(values):
        for col_offset, cell in enumerate(row):
            grid[row1 - 1 + row_offset][col1 - 1 + col_offset] = str(cell)
    book[sheet] = grid
    _save(book)
    return {
        "sheet": sheet,
        "cell_range": cell_range.strip().upper(),
        "updated": expected_rows * expected_cols,
    }


@mcp.tool(annotations=_WRITE)
def append_row(sheet: str, values: list[str]) -> dict:
    """Append one row to the end of a sheet."""
    _check_sheet(sheet)
    if not values:
        raise ValueError("values is empty")
    _flake()
    book = _workbook()
    grid = book.setdefault(sheet, [])
    grid.append([str(cell) for cell in values])
    book[sheet] = grid
    _save(book)
    return {"sheet": sheet, "row": len(grid), "values": grid[-1]}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
