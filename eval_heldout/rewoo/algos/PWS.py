# main class chaining Planner, Worker and Solver.
import re
import time

from nodes.Planner import Planner
from nodes.Solver import Solver
from nodes.Worker import *
from utils.util import *


class PWS:
    def __init__(self, available_tools=["Google", "LLM"], fewshot="\n", planner_model="text-davinci-003",
                 solver_model="text-davinci-003"):
        self.workers = available_tools
        self.planner = Planner(workers=self.workers,
                               model_name=planner_model,
                               fewshot=fewshot)
        self.solver = Solver(model_name=solver_model)
        self.plans = []
        self.planner_evidences = {}
        self.worker_evidences = {}
        self.tool_counter = {}
        self.planner_token_unit_price = get_token_unit_price(planner_model)
        self.solver_token_unit_price = get_token_unit_price(solver_model)
        self.tool_token_unit_price = get_token_unit_price("text-davinci-003")
        self.google_unit_price = 0.01

    # input: the question line. e.g. "Question: What is the capital of France?"
    def run(self, input):
        # run is stateless, so we need to reset the evidences
        self._reinitialize()
        result = {}
        st = time.time()
        # Plan
        planner_response = self.planner.run(input, log=True)
        plan = planner_response["output"]
        planner_log = planner_response["input"] + planner_response["output"]
        self.plans = self._parse_plans(plan)
        self.planner_evidences = self._parse_planner_evidences(plan)
        #assert len(self.plans) == len(self.planner_evidences)

        # Work
        self._get_worker_evidences()
        worker_log = ""
        for i in range(len(self.plans)):
            e = f"#E{i + 1}"
            worker_log += f"{self.plans[i]}\nEvidence:\n{self.worker_evidences[e]}\n"

        # Solve
        solver_response = self.solver.run(input, worker_log, log=True)
        output = solver_response["output"]
        solver_log = solver_response["input"] + solver_response["output"]

        result["wall_time"] = time.time() - st
        result["input"] = input
        result["output"] = output
        result["planner_log"] = planner_log
        result["worker_log"] = worker_log
        result["solver_log"] = solver_log
        result["tool_usage"] = self.tool_counter
        result["steps"] = len(self.plans) + 1
        result["total_tokens"] = planner_response["prompt_tokens"] + planner_response["completion_tokens"] \
                                 + solver_response["prompt_tokens"] + solver_response["completion_tokens"] \
                                 + self.tool_counter.get("LLM_token", 0) \
                                 + self.tool_counter.get("Calculator_token", 0)
        result["token_cost"] = self.planner_token_unit_price * (planner_response["prompt_tokens"] + planner_response["completion_tokens"]) \
                               + self.solver_token_unit_price * (solver_response["prompt_tokens"] + solver_response["completion_tokens"]) \
                               + self.tool_token_unit_price * (self.tool_counter.get("LLM_token", 0) + self.tool_counter.get("Calculator_token", 0))
        result["tool_cost"] = self.tool_counter.get("Google", 0) * self.google_unit_price
        result["total_cost"] = result["token_cost"] + result["tool_cost"]

        return result

    def _parse_plans(self, response):
        plans = []
        for line in response.splitlines():
            if line.startswith("Plan:"):
                plans.append(line)
        return plans

    def _parse_planner_evidences(self, response):
        evidences = {}
        for line in response.splitlines():
            if line.startswith("#") and line[1] == "E" and line[2].isdigit():
                e, tool_call = line.split("=", 1)
                e, tool_call = e.strip(), tool_call.strip()
                if len(e) == 3:
                    evidences[e] = tool_call
                else:
                    evidences[e] = "No evidence found"
        return evidences

    # use planner evidences to assign tasks to respective workers.
    def _get_worker_evidences(self):
        for e, tool_call in self.planner_evidences.items():
            if "[" not in tool_call:
                self.worker_evidences[e] = tool_call
                continue
            tool, tool_input = tool_call.split("[", 1)
            tool_input = tool_input[:-1]
            # find variables in input and replace with previous evidences
            for var in re.findall(r"#E\d+", tool_input):
                if var in self.worker_evidences:
                    tool_input = tool_input.replace(var, "[" + self.worker_evidences[var] + "]")
            if tool in self.workers:
                self.worker_evidences[e] = WORKER_REGISTRY[tool].run(tool_input)
                if tool == "Google":
                    self.tool_counter["Google"] = self.tool_counter.get("Google", 0) + 1  # number of query
                elif tool == "LLM":
                    self.tool_counter["LLM_token"] = self.tool_counter.get("LLM_token", 0) + len(
                        tool_input + self.worker_evidences[e]) // 4
                elif tool == "Calculator":
                    self.tool_counter["Calculator_token"] = self.tool_counter.get("Calculator_token", 0) \
                                                            + len(
                        LLMMathChain(llm=OpenAI(), verbose=False).prompt.template + tool_input + self.worker_evidences[
                            e]) // 4
            else:
                self.worker_evidences[e] = "No evidence found"

    def _reinitialize(self):
        self.plans = []
        self.planner_evidences = {}
        self.worker_evidences = {}
        self.tool_counter = {}


class PWS_Base(PWS):
    def __init__(self, fewshot=fewshots.HOTPOTQA_PWS_BASE, planner_model="text-davinci-003",
                 solver_model="text-davinci-003", available_tools=["Wikipedia", "LLM"]):
        super().__init__(available_tools=available_tools,
                         fewshot=fewshot,
                         planner_model=planner_model,
                         solver_model=solver_model)


class PWS_Extra(PWS):
    def  __init__(self, fewshot=fewshots.HOTPOTQA_PWS_EXTRA, planner_model="text-davinci-003",
                 solver_model="text-davinci-003", available_tools=["Google", "Calculator", "LLM"]):
        super().__init__(available_tools=available_tools,
                         fewshot=fewshot,
                         planner_model=planner_model,
                         solver_model=solver_model)
