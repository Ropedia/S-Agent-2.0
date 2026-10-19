from sagent.agents.base import BaseAgent
from sagent.environments.base import BaseEnvironment
from sagent.models.agent.context import AgentContext
from sagent.models.agent.name import AgentName


class NopAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return AgentName.NOP.value

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        pass
