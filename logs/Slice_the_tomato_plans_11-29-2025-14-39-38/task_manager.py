import threading
import time
import uuid
import random
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, List


class TaskStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Task:
    id: str
    name: str
    payload: dict = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None  # agent_id
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class TaskManager:
    """
    Central Task Manager managing per-agent tasks.
    - Each agent can have multiple tasks in RUNNING state at the same time.
    - Newer tasks are treated with higher priority (LIFO / stack behavior).
    - All accesses are protected by a Lock for thread safety.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._tasks: Dict[str, Task] = {}  # task_id -> Task
        # For each agent, keep a list (stack) of currently running task_ids
        self._agent_running_tasks: Dict[str, List[str]] = {}  # agent_id -> [task_id, ...]

    # ---------- Agent management ----------

    def register_agent(self, agent_id: str):
        with self._lock:
            if agent_id not in self._agent_running_tasks:
                self._agent_running_tasks[agent_id] = []
                print(f"[TaskManager] Registered agent {agent_id}")

    # ---------- Task registration ----------

    def add_task(self, agent_id: str, name: str, payload: dict = None) -> str:
        """
        Add a task for the specific agent.
        - The task is immediately assigned to the agent.
        - The task status becomes RUNNING.
        - The task is pushed to the end of the agent's running stack.
          (So newer tasks are processed first: LIFO)
        """
        if payload is None:
            payload = {}

        with self._lock:
            if agent_id not in self._agent_running_tasks:
                raise ValueError(f"Agent {agent_id} is not registered")

            task_id = str(uuid.uuid4().hex[:8])
            task = Task(id=task_id, name=name, payload=payload)
            task.status = TaskStatus.RUNNING
            task.assigned_to = agent_id
            task.updated_at = time.time()

            self._tasks[task_id] = task
            self._agent_running_tasks[agent_id].append(task_id)

            print(f"[TaskManager] Added & started task {task_id} ({name}) for {agent_id}")
            return task_id

    # ---------- Task retrieval & completion ----------

    def get_next_task(self, agent_id: str) -> Optional[Task]:
        """
        Get the task that the agent should work on now.
        - If the agent has running tasks, return the most recently added one (top of stack).
        - If there is no running task for this agent, return None.
        """
        with self._lock:
            if agent_id not in self._agent_running_tasks:
                raise ValueError(f"Agent {agent_id} is not registered")

            running_stack = self._agent_running_tasks[agent_id]
            if not running_stack:
                return None

            # Newest task is at the end of the list (stack top)
            top_task_id = running_stack[-1]
            return self._tasks.get(top_task_id)

    def complete_task(
        self,
        agent_id: str = None,
        task_id: str = None,
        task_name: str = None,
        success: bool = True
    ):
        """
        Mark a task as completed.
        This function supports multiple ways to identify a task:
        1. By task_id (direct)
        2. By (agent_id, task_name):
           - It searches the agent's running stack and completes
             the last task whose name matches task_name.
        """
        with self._lock:

            # --- Case 1: Identify by task_id (direct) ---
            if task_id is not None:
                task = self._tasks.get(task_id)
                if not task:
                    print(f"[TaskManager] complete_task: invalid task_id={task_id}")
                    return
                agent = task.assigned_to

            # --- Case 2: Identify by (agent_id, task_name) ---
            else:
                if agent_id is None or task_name is None:
                    raise ValueError("Either task_id OR (agent_id AND task_name) must be provided")

                if agent_id not in self._agent_running_tasks:
                    print(f"[TaskManager] Agent {agent_id} is not registered")
                    return

                running_stack = self._agent_running_tasks[agent_id]
                if not running_stack:
                    print(f"[TaskManager] No running tasks for agent {agent_id}")
                    return

                # Search from the end (newest first) for a matching task name
                found_task_id = None
                for tid in reversed(running_stack):
                    t = self._tasks.get(tid)
                    if t and t.name == task_name:
                        found_task_id = tid
                        break

                if found_task_id is None:
                    print(f"[TaskManager] No running task with name '{task_name}' for agent {agent_id}")
                    return

                task_id = found_task_id
                task = self._tasks[task_id]
                agent = agent_id

            # --- Mark the task as completed ---
            task.status = TaskStatus.DONE if success else TaskStatus.FAILED
            task.updated_at = time.time()

            # Remove from the agent's running stack if present
            if agent is not None and agent in self._agent_running_tasks:
                if task_id in self._agent_running_tasks[agent]:
                    self._agent_running_tasks[agent].remove(task_id)

            print(f"[TaskManager] Task {task_id} ({task.name}) is "
                  f"{'DONE' if success else 'FAILED'} (agent={agent})")

    # ---------- Status query ----------

    def get_agent_current_tasks(self, agent_id: str) -> List[Task]:
        """
        Return all tasks currently in RUNNING state for this agent.
        (In stack order: older first, newer last.)
        """
        with self._lock:
            if agent_id not in self._agent_running_tasks:
                raise ValueError(f"Agent {agent_id} is not registered")

            ids = list(self._agent_running_tasks[agent_id])
            return [self._tasks[tid] for tid in ids]

    def get_agent_top_task(self, agent_id: str) -> Optional[Task]:
        """
        Convenience helper:
        Return the top (most recently added) running task for this agent.
        """
        with self._lock:
            if agent_id not in self._agent_running_tasks:
                raise ValueError(f"Agent {agent_id} is not registered")
            if not self._agent_running_tasks[agent_id]:
                return None
            tid = self._agent_running_tasks[agent_id][-1]
            return self._tasks.get(tid)

    def find_idle_agents(self) -> List[str]:
        """
        Return a list of agent_ids that currently have no running tasks.
        """
        with self._lock:
            idle_agents = [
                agent_id
                for agent_id, stack in self._agent_running_tasks.items()
                if not stack  # empty list => no running tasks
            ]
            return idle_agents   

    def debug_print(self):
        with self._lock:
            print("========== TaskManager State ==========")
            print("Agents & running tasks (stack, older -> newer):")
            for agent_id, stack in self._agent_running_tasks.items():
                print(f"  {agent_id}: {stack}")
            print("\nAll tasks:")
            for t in self._tasks.values():
                print(f"  {t.id} | {t.name} | {t.status.name} | assigned_to={t.assigned_to}")
            print("=======================================")


# ---------- Worker thread (robot) example ----------

def agent_worker(agent_id: str, manager: TaskManager):
    manager.register_agent(agent_id)

    while True:
        task = manager.get_next_task(agent_id)
        if task is None:
            print(f"[{agent_id}] No running tasks. Stopping.")
            break

        print(f"[{agent_id}] Work on task {task.id} ({task.name}) "
              f"payload={task.payload}")

        # Actual robot work (e.g., AI2-THOR controller.step...) goes here
        work_time = random.uniform(0.3, 0.8)
        time.sleep(work_time)

        # For demo, we always complete the top task
        manager.complete_task(task_id=task.id, success=True)
        print(f"[{agent_id}] Finished task {task.id} (OK)")

        # Small pause before checking next task
        time.sleep(0.1)


if __name__ == "__main__":
    manager = TaskManager()

    # Example: each agent gets multiple tasks, some of which conceptually depend on others.
    agent_ids = [f"agent_{i}" for i in range(2)]
    for aid in agent_ids:
        manager.register_agent(aid)
        # Example of hierarchical / related tasks:
        # High-level task
        manager.add_task(
            agent_id=aid,
            name="PickupApple",
            payload={"agent": aid, "object": "Apple"}
        )
        # Subtask registered later → should be processed first (LIFO)
        manager.add_task(
            agent_id=aid,
            name="GoToApple",
            payload={"agent": aid, "target": "AppleLocation"}
        )
        # Another task registered even later
        manager.add_task(
            agent_id=aid,
            name="AvoidCollision",
            payload={"agent": aid}
        )

    # Launch threads
    threads = []
    for aid in agent_ids:
        t = threading.Thread(target=agent_worker, args=(aid, manager), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    manager.debug_print()
