"""Shared result format for tools."""


def ok(data: dict) -> dict:
    return {"ok": True, "data": data}


def error(code: str, message: str) -> dict:
    return {"ok": False, "error": code, "message": message}