from __future__ import annotations

import io
from collections import deque
from dataclasses import dataclass
from typing import Callable, Any, Set

from frozendict import frozendict
from frozenlist2 import frozenlist


@dataclass(frozen=True)
class Memory:
    vars: frozendict[str, Any]

    def set(self, **kwargs):
        new_vars = dict(self.vars)
        for var, value in kwargs.items():
            new_vars[var] = value
        return Memory(frozendict(new_vars))

    def get(self, key):
        return self.vars.get(key)


@dataclass(frozen=True)
class Thread:
    local_memory: Memory
    states: frozenlist[Callable[[Memory, Memory], (Memory, Memory, int, bool)]]
    name: str
    current_state: int = 0
    is_running: bool = True

    def run(self, shared_memory) -> (Thread, Memory):
        new_shared_memory, new_local_memory, new_state, is_running \
            = self.states[self.current_state](shared_memory, self.local_memory)
        return Thread(new_local_memory, self.states, self.name, new_state, is_running), new_shared_memory


@dataclass(frozen=True)
class GlobalState:
    shared_memory: Memory
    threads: frozenlist[Thread]

    def run(self) -> Set[GlobalState]:
        result = set()
        for i in range(len(self.threads)):
            thread = self.threads[i]
            if not thread.is_running:
                continue
            new_state, new_shared_memory = thread.run(self.shared_memory)
            new_thread_pool = self.threads.copy()
            new_thread_pool[i] = new_state
            result.add(GlobalState(new_shared_memory, frozenlist(new_thread_pool)))
        return result

    def __str__(self):
        variables_names = list(self.shared_memory.vars.keys())
        variables_names.sort()
        variable_values = [self.shared_memory.vars[i] for i in variables_names]
        result = "["
        for i in self.threads:
            result += print_to_string(i.name, i.current_state, sep='', end=',')
        result += print_to_string(*variable_values, sep=',', end=']')
        return result


@dataclass(frozen=True)
class Transition:
    start: GlobalState
    end: GlobalState

    def __str__(self):
        result = str(self.start)
        result += ' -> '
        result += str(self.end)
        return result


def print_to_string(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents


def nop(shared, local):
    return shared, local, 1


def main():
    used = set()
    queue = deque()

    queue.append(GlobalState(shared_memory=Memory(frozendict(a=0, b=0)),
                             threads=frozenlist([
                                 Thread(current_state=1,
                                        name="P",
                                        local_memory=Memory(frozendict()),
                                        states=frozenlist([
                                            nop,  # 0
                                            lambda shared, local: (shared.set(a=1), local, 2, True),  # 1
                                            lambda shared, local: (shared, local, 2, True) if shared.get(
                                                'b') != 0 else (shared, local, 3, True),  # 2
                                            lambda shared, local: (shared, local, 4, True),  # 3
                                            lambda shared, local: (shared.set(a=0), local, 1, True),  # 4
                                        ])),
                                 Thread(current_state=1,
                                        name="Q",
                                        local_memory=Memory(frozendict()),
                                        states=frozenlist([
                                            nop,  # 0
                                            lambda shared, local: (shared.set(b=1), local, 2, True),  # 1
                                            lambda shared, local: (shared, local, 4, False) if shared.get(
                                                'a') == 0 else (shared, local, 3, True),  # 2
                                            lambda shared, local: (shared.set(b=0), local, 1, True),  # 3
                                        ])),
                             ])))

    transitions = set()
    while len(queue) > 0:
        item = queue.popleft()
        new_states = item.run()

        for i in new_states:
            transitions.add(Transition(item, i))

        clean_new_states = new_states.difference(used)
        used.update(clean_new_states)
        queue.extend(clean_new_states)
        print(f"Processing {len(queue)} items")

    print(f"Done, got {len(used)} possible states, {len(transitions)} possible transitions")
    print()
    print(*used, sep='\n')
    print()
    print(*transitions, sep='\n')


if __name__ == "__main__":
    main()
