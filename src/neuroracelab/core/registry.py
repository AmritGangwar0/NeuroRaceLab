from dataclasses import dataclass


@dataclass(frozen=True)
class RegistryEntry:
    identifier: str
    factory: object
    metadata: dict
class Registry:
    def __init__(self): self._entries={}
    def register(self, identifier, factory, **metadata):
        if identifier in self._entries: raise ValueError(f"Duplicate registry identifier: {identifier}")
        self._entries[identifier]=RegistryEntry(identifier,factory,metadata)
    def get(self, identifier):
        if identifier not in self._entries: raise KeyError(f"Unknown registry identifier: {identifier}")
        return self._entries[identifier]
    def create(self, identifier, **kwargs): return self.get(identifier).factory(**kwargs)
    def identifiers(self): return tuple(sorted(self._entries))
    def metadata(self, identifier): return dict(self.get(identifier).metadata)
